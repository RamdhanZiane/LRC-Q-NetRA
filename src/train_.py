# for parsing input command
import argparse
import os
import time
import math
import numpy as np
import random

# for logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# import utils for data preparation and algorithmic models
from utils import to_gpu, Corpus, batchify, generate_walks
from models import Seq2Seq, MLP_D, MLP_G

# import visualization module
from viz_karate import viz

from LRC import generate_LRC

EMBED_SEGMENT = 4000

# import networkx package for graph data input parse, need to install networkx by: pip install networkx
import networkx as nx

# import scipy package for geting laplacian of graph, need to install networkx by: pip install scipy
from scipy.sparse import csgraph

def train(A_nx=None, dictionary=None):
    """Parameters to parse
            Path Arguments: The input and output directory
            Data Processing Arguments: data preprocessing for generating ``walks'' from the graph
            Model Arguments: parameters for the model
            Training Arguments, Evaluation Arguments, and others like 
    """
    if dictionary:
        args = argparse.Namespace(**dictionary)
    else:
        parser = argparse.ArgumentParser(description='NetRA')
        parser.add_argument('--data_path', type=str, default='../data/karate.adjlist',
                            help='location of the data corpus')                        # location of the graph with linked list format
        parser.add_argument('--outf', type=str, default='example',
                            help='output directory name')                              # location of output embeddings in different epochs

        # Data Processing Arguments
        parser.add_argument('--maxlen', type=int, default=100,
                            help='maximum sentence length')                            # the parameter is for random walk to generating walks,
                                                                                    # this is the upbound of the walk length, in the code we generate walks with the same length


        # Model Arguments
        ################### important hyper-parameters ################################
        parser.add_argument('--nhidden', type=int, default=2,
                            help='number of hidden units per layer')                   # dimension of embedding vectors, since we want to visualize to 2-dimensional
        parser.add_argument('--emsize', type=int, default=30,
                            help='size of word embeddings')                            # large graph 100-300, this is the size of input after original one hot embedding's mapping


        ################### typically below are set to default ones ###################
        parser.add_argument('--nlayers', type=int, default=1,
                            help='number of layers')                                   # number of stacked LSTM for autoencoding
        parser.add_argument('--noise_radius', type=float, default=0.2,
                            help='stdev of noise for autoencoder (regularizer)')       # stard deviation of noise for autoencoder
        parser.add_argument('--noise_anneal', type=float, default=0.995,
                            help='anneal noise_radius exponentially by this'
                                'every 100 iterations')                               # decay rate for exponentially decaying noise on autoencoder
        parser.add_argument('--hidden_init', action='store_true',
                            help="initialize decoder hidden state with encoder's")
        parser.add_argument('--arch_g', type=str, default='300-300',
                            help='generator architecture (MLP)')                       # specify the MLP structure of generator in GAN;
                                                                                    # for example, 300-300 means two layers, each layer includes 300 nodes
        parser.add_argument('--arch_d', type=str, default='300-300',
                            help='critic/discriminator architecture (MLP)')            # specify the MLP structure of discriminator in GAN;
                                                                                    # for example, 300-300 means two layers, each layer includes 300 nodes
        parser.add_argument('--z_size', type=int, default=100,
                            help='dimension of random noise z to feed into generator') # random noise to be feed into the generator
        parser.add_argument('--temp', type=float, default=1,
                            help='softmax temperature (lower --> more discrete)')      # specify the temperature of softmax, \tau
        parser.add_argument('--enc_grad_norm', type=bool, default=True,
                            help='norm code gradient from critic->encoder')
        parser.add_argument('--gan_toenc', type=float, default=-0.01,
                            help='weight factor passing gradient from gan to encoder') # weight factor passing from gradient of GAN to encoder, thi is used by grad_hook
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='dropout applied to layers (0 = no dropout)')         # dropout to prevent overfitting, by default, there is no dropout


        # Training Arguments
        ################### important hyper-parameters ################################
        parser.add_argument('--epochs', type=int, default=50,
                            help='maximum number of epochs')                           # epochs for training, usually small graph 50, large graph 100
        parser.add_argument('--walk_length', type=int, default=20,
                            help='length of walk sampled from the graph')              # the length of walk sampled rooted from each node
        parser.add_argument('--numWalks_per_node', type=int, default=30,
                            help='number of walks sampled for each node')              # number of walks sampled for each node
        parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                            help='batch size')                                         # batch size for training
        parser.add_argument('--niters_ae', type=int, default=1,
                            help='number of autoencoder iterations in training')       # in each epoch, number of iterations for training autoencoder
        parser.add_argument('--niters_gan_d', type=int, default=5,
                            help='number of discriminator iterations in training')     # in each epoch, number of iterations for training discriminator
        parser.add_argument('--niters_gan_g', type=int, default=1,
                            help='number of generator iterations in training')         # in each epoch, number of iterations for training generator

        parser.add_argument('--niters_gan_schedule', type=str, default='2-4-6-10-20-30-40',
                            help='epoch counts to increase number of GAN training '
                                ' iterations (increment by 1 each time)')             # in different epochs, dynamically increase the GAN iterations,
                                                                                    # for example, 2-4-6 means, 2 epochs then increase one, 4 epochs then increase again

        ################### typically below are set to default ones ###################
        parser.add_argument('--min_epochs', type=int, default=6,
                            help="minimum number of epochs to train for")              # minimum nuber of epochs for training
        parser.add_argument('--no_earlystopping', action='store_true',
                            help="won't use KenLM for early stopping")                 # if conduct early stopping
        parser.add_argument('--lr_ae', type=float, default=1,
                            help='autoencoder learning rate')                          # learning rate for AE, because it is using SDG, by default it is 1
        parser.add_argument('--lr_gan_g', type=float, default=5e-05,
                            help='generator learning rate')                            # learning rate for generator, because it is using ADM, by default it is a smaller one
        parser.add_argument('--lr_gan_d', type=float, default=1e-05,
                            help='critic/discriminator learning rate')                 # learning rate for discriminator, because it is using ADM, by default it is a smaller one
        parser.add_argument('--beta1', type=float, default=0.9,
                            help='beta1 for adam. default=0.9')                        # beta for adam
        parser.add_argument('--clip', type=float, default=1,
                            help='gradient clipping, max norm')                        # gradient clipping
        parser.add_argument('--gan_clamp', type=float, default=0.01,
                            help='WGAN clamp')                                         # WGAN clamp

        # Evaluation Arguments
        parser.add_argument('--sample', action='store_true',
                            help='sample when decoding for generation')
        parser.add_argument('--log_interval', type=int, default=200,
                            help='interval to log autoencoder training results')

        # Other
        parser.add_argument('--seed', type=int, default=1111,
                            help='random seed')                                        # random seeds for parameter initialization
        parser.add_argument('--cuda', action='store_true',
                            help='use CUDA')                                           # use CUDA for training

        args = parser.parse_args()
        args_dict = vars(args)


        # Dump dictionary to JSON file
        with open('testargs.json', 'w') as f:
            json.dump(args_dict, f, indent=4)
    print(vars(args))


    # make output directory if it doesn't already exist
    if not os.path.isdir('./output'):
        os.makedirs('./output')
    if not os.path.isdir('./output/{}'.format(args.outf)):
        os.makedirs('./output/{}'.format(args.outf))


    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, "
                "so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)


    ###############################################################################
    # Load data
    ###############################################################################


    print(os.getcwd())
    # A_nx = nx.read_edgelist(args.data_path,nodetype = int, create_using=nx.DiGraph())  # if the graph is with sparse edge format, each line one edge, then use this
    if not A_nx:
        A_nx = nx.read_adjlist('../data/karate.adjlist', nodetype=int)                        # use this for reading adjacent list format graph
    A_nx = generate_LRC(A_nx,False)

    A = nx.to_scipy_sparse_array(A_nx)                                         # transfer to sparse matrix format
    L = csgraph.laplacian(A, normed=False)                                      # use csgraph package to calculate the laplacian
    L = np.array(L.toarray(), np.float32)
    L = Variable(torch.from_numpy(L))
    L = to_gpu(args.cuda, L)





    # generate walk for each node with given walk_length
    walks = generate_walks(A_nx, args.numWalks_per_node, args.walk_length)
    # save randomly generated walks to file ../tmp/train.txt

    np.savetxt('./tmp/train.txt', walks, delimiter=" ", fmt="%s")

    # create corpus
    corpus = Corpus('./tmp/',
                    maxlen=args.maxlen)
    # dumping vocabulary
    with open('./output/{}/vocab.json'.format(args.outf), 'w') as f:
        json.dump(corpus.dictionary.word2idx, f)

    # save arguments to args.json and logs.txt
    ntokens = len(corpus.dictionary.word2idx)
    print("Vocabulary Size: {}".format(ntokens))
    args.ntokens = ntokens
    with open('./output/{}/args.json'.format(args.outf), 'w') as f:
        json.dump(vars(args), f)
    with open("./output/{}/logs.txt".format(args.outf), 'w') as f:
        f.write(str(vars(args)))
        f.write("\n\n")

    # preparing batches for training
    train_data = batchify(corpus.train, args.batch_size, shuffle=True)

    print("Loaded data!")

    ###############################################################################
    # Build the models
    ###############################################################################

    ntokens = len(corpus.dictionary.word2idx)
    autoencoder = Seq2Seq(emsize=args.emsize,
                        nhidden=args.nhidden,
                        ntokens=ntokens,
                        nlayers=args.nlayers,
                        noise_radius=args.noise_radius,
                        hidden_init=args.hidden_init,
                        dropout=args.dropout,
                        gpu=args.cuda)

    gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
    gan_disc = MLP_D(ninput=args.nhidden, noutput=1, layers=args.arch_d)

    print(autoencoder)
    print(gan_gen)
    print(gan_disc)


    #### optimizing AE, GAN-generator, GAN-discriminator
    ## SGD, learning rate should be larger, like 1, Adam's learning rate should be smaller, like 0.001
    optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
    optimizer_gan_g = optim.Adam(gan_gen.parameters(),
                                lr=args.lr_gan_g,
                                betas=(args.beta1, 0.999))
    optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                                lr=args.lr_gan_d,
                                betas=(args.beta1, 0.999))
    #### crossEntropy loss for discriminator
    criterion_ce = nn.CrossEntropyLoss()

    if args.cuda:
        autoencoder = autoencoder.cuda()
        gan_gen = gan_gen.cuda()
        gan_disc = gan_disc.cuda()
        criterion_ce = criterion_ce.cuda()


    ###############################################################################
    # Training code
    ###############################################################################


    def save_model():
        print("Saving models")
        with open('./output/{}/autoencoder_model.pt'.format(args.outf), 'wb') as f:
            torch.save(autoencoder.state_dict(), f)
        with open('./output/{}/gan_gen_model.pt'.format(args.outf), 'wb') as f:
            torch.save(gan_gen.state_dict(), f)
        with open('./output/{}/gan_disc_model.pt'.format(args.outf), 'wb') as f:
            torch.save(gan_disc.state_dict(), f)


    def embed_afterLSTM(corpus, emsize):
        """
        Getting embedding codes
        :param corpus: nodes of graph
        :param emsize: number of dimensions of embedded codes
        :return: embedding vectors
        """
        dic = corpus.dictionary.word2idx.values()
        dic = np.sort(list(dic))
        if ntokens <= EMBED_SEGMENT:
            dic = np.vstack((dic,dic))
            dic_to_embed = Variable(torch.from_numpy(dic))
            dic_to_embed = to_gpu(args.cuda, dic_to_embed)
            embeded = autoencoder.embed_after_LSTM(dic_to_embed, [ntokens,ntokens])
            dic_vector = embeded[:ntokens].cpu().data.numpy()
        else:
            dic_i = np.array_split(dic, ntokens/EMBED_SEGMENT)
            dic_1 = dic_i[0]
            n_dic_1 = len(dic_1)
            dic_1 = np.vstack((dic_1,dic_1))
            dic_to_embed = Variable(torch.from_numpy(dic_1))
            dic_to_embed = to_gpu(args.cuda, dic_to_embed)
            embeded = autoencoder.embed_after_LSTM(dic_to_embed, [n_dic_1,n_dic_1])
            dic_vector = embeded[:n_dic_1].cpu().data.numpy()
            for j in range(1, ntokens/EMBED_SEGMENT):
                dic_j = dic_i[j]
                n_dic_j = len(dic_j)
                dic_j = np.vstack((dic_j,dic_j))
                dic_to_embed = Variable(torch.from_numpy(dic_j))
                dic_to_embed = to_gpu(args.cuda, dic_to_embed)
                embeded = autoencoder.embed_after_LSTM(dic_to_embed, [n_dic_j,n_dic_j])
                dic_vector_j = embeded[:n_dic_j].cpu().data.numpy()
                dic_vector = np.vstack((dic_vector, dic_vector_j))
            dic = np.vstack((dic,dic))
                
        dic_tosave = np.insert(dic_vector, 0, dic[0], axis=1)


        np.savetxt('./output/{}/embed_afterLSTM_{}.txt'.format(args.outf, epoch), dic_tosave,
                fmt=' '.join(['%i'] + ['%1.6f'] * emsize))
        return dic_vector

    def embed_dic_for_loss(corpus, emsize):
        dic = corpus.dictionary.word2idx.values()
        dic = np.sort(list(dic))

        if ntokens <= EMBED_SEGMENT:
            dic = np.vstack((dic,dic))
            dic_to_embed = Variable(torch.from_numpy(dic))
            dic_to_embed = to_gpu(args.cuda, dic_to_embed)
            embeded = autoencoder.embed_after_LSTM(dic_to_embed, [ntokens,ntokens])
            dic_vector = embeded[:ntokens]
        else:
            dic_i = np.array_split(dic, ntokens/EMBED_SEGMENT)
            dic_1 = dic_i[0]
            n_dic_1 = len(dic_1)
            dic_1 = np.vstack((dic_1,dic_1))
            dic_to_embed = Variable(torch.from_numpy(dic_1))
            dic_to_embed = to_gpu(args.cuda, dic_to_embed)
            embeded = autoencoder.embed_after_LSTM(dic_to_embed, [n_dic_1,n_dic_1])
            dic_vector = embeded[:n_dic_1]
            for j in range(1, ntokens/EMBED_SEGMENT):
                dic_j = dic_i[j]
                n_dic_j = len(dic_j)
                dic_j = np.vstack((dic_j,dic_j))
                dic_to_embed = Variable(torch.from_numpy(dic_j))
                dic_to_embed = to_gpu(args.cuda, dic_to_embed)
                embeded = autoencoder.embed_after_LSTM(dic_to_embed, [n_dic_j,n_dic_j])
                dic_vector_j = embeded[:n_dic_j]
                dic_vector = torch.cat((dic_vector, dic_vector_j), 0)
        return dic_vector


    def unique(tensor1d):
        t, idx = np.unique(tensor1d.cpu().data.numpy(), return_index=True)
        return t, idx


    def train_ae(batch, total_loss_ae, start_time, i):
        """
        Training LSTM AE
        :param batch: one batch of data
        :param total_loss_ae: accumulated loss for LSTM AE so far
        :param start_time: for timming
        :param i: current iteration ID
        :return: accumulated total loss of LSTM AE part so far, and start time for timing
        """
        autoencoder.train()
        autoencoder.zero_grad()

        source, target, lengths = batch
        source = to_gpu(args.cuda, Variable(source))
        target = to_gpu(args.cuda, Variable(target))

        # Create sentence length mask over padding
        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

        # output: batch x seq_len x ntokens
        output = autoencoder(source, lengths, noise=True)

        # output_size: batch_size, maxlen, self.ntokens
        flattened_output = output.view(-1, ntokens)


        emb_py = embed_dic_for_loss(corpus, args.nhidden)
        embT = torch.transpose(emb_py,0,1)
        embT = torch.mm(embT, L)

        adj_loss = torch.trace(torch.mm(embT, emb_py)) / ntokens

        masked_output = \
            flattened_output.masked_select(output_mask).view(-1, ntokens)
        loss = criterion_ce(masked_output/args.temp, masked_target)
        print('loss', loss)

        loss += adj_loss

        loss.backward()

        # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
        # This is the version of Wasserstein GAN, which has gradient clipping
        torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
        optimizer_ae.step()

        total_loss_ae += loss.data

        accuracy = None

        ######################## store log periodically ############################
        if i % args.log_interval == 0 and i > 0:
            # accuracy
            probs = F.softmax(masked_output)
            max_vals, max_indices = torch.max(probs, 1)
            accuracy = torch.mean(max_indices.eq(masked_target).float()).data[0]

            cur_loss = total_loss_ae[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'
                .format(epoch, i, len(train_data),
                        elapsed * 1000 / args.log_interval,
                        cur_loss, math.exp(cur_loss), accuracy))

            with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                f.write('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}\n'.
                        format(epoch, i, len(train_data),
                            elapsed * 1000 / args.log_interval,
                            cur_loss, math.exp(cur_loss), accuracy))

            total_loss_ae = 0
            start_time = time.time()

        return total_loss_ae, start_time


    def train_gan_g():
        """
        Training WGAN generator network
        :return: error of generator part
        """
        gan_gen.train()
        gan_gen.zero_grad()

        noise = to_gpu(args.cuda,
                    Variable(torch.ones(args.batch_size, args.z_size)))
        noise.data.normal_(0, 1)

        fake_hidden = gan_gen(noise)
        errG = gan_disc(fake_hidden)

        # loss / backprop
        errG.backward(one)
        optimizer_gan_g.step()

        return errG


    def grad_hook(grad):
        # Gradient norm: regularize to be same
        # code_grad_gan * code_grad_ae / norm(code_grad_gan)
        if args.enc_grad_norm:
            gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
            normed_grad = grad * autoencoder.grad_norm / gan_norm
        else:
            normed_grad = grad

        # weight factor and sign flip
        normed_grad *= -math.fabs(args.gan_toenc)
        return normed_grad


    def train_gan_d(batch):
        """
        Training WGAN discriminator
        :param batch: training batch data
        :return: discriminator part error
        """
        # clamp parameters to a cube
        # WGAN Weight clipping
        for p in gan_disc.parameters():
            p.data.clamp_(-args.gan_clamp, args.gan_clamp)

        autoencoder.train()
        autoencoder.zero_grad()
        gan_disc.train()
        gan_disc.zero_grad()

        # positive samples ----------------------------
        # generate real codes
        source, target, lengths = batch
        source = to_gpu(args.cuda, Variable(source))
        target = to_gpu(args.cuda, Variable(target))

        # batch_size x nhidden
        real_hidden = autoencoder(source, lengths, noise=False, encode_only=True)
        real_hidden.register_hook(grad_hook)

        # loss / backprop
        errD_real = gan_disc(real_hidden)
        errD_real.backward(one)

        # negative samples ----------------------------
        # generate fake codes
        noise = to_gpu(args.cuda,
                    Variable(torch.ones(args.batch_size, args.z_size)))
        noise.data.normal_(0, 1)

        # loss / backprop
        fake_hidden = gan_gen(noise)
        errD_fake = gan_disc(fake_hidden.detach())
        errD_fake.backward(mone)

        # `clip_grad_norm` to prvent exploding gradient problem in RNNs / LSTMs
        # This is the version of Wasserstein GAN, which has gradient clipping
        torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)

        optimizer_gan_d.step()
        optimizer_ae.step()
        errD = -(errD_real - errD_fake)

        return errD, errD_real, errD_fake


    print("Training...")
    with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
        f.write('Training...\n')

    # schedule of increasing GAN training loops
    if args.niters_gan_schedule != "":
        gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
    else:
        gan_schedule = []
    niter_gan = 1   # start from 1, and will be dynamically increased

    fixed_noise = to_gpu(args.cuda,
                        Variable(torch.ones(args.batch_size, args.z_size)))
    fixed_noise.data.normal_(0, 1)
    one = to_gpu(args.cuda, torch.FloatTensor([1]))
    one = torch.tensor(1)
    mone = one * -1

    for epoch in range(1, args.epochs+1):
        # embed_afterLSTM(corpus, args.nhidden)

        # update gan training schedule
        if epoch in gan_schedule:
            niter_gan += 1
            print("GAN training loop schedule increased to {}".format(niter_gan))
            with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                f.write("GAN training loop schedule increased to {}\n".
                        format(niter_gan))

        total_loss_ae = 0
        epoch_start_time = time.time()
        start_time = time.time()
        niter = 0
        niter_global = 1

        # loop through all batches in training data
        while niter < len(train_data):

            """ 
                Iteratively conduct autoencoder training, then GAN regularization,
                The GAN part includes discriminator and generator iteratively.
            """

            # train autoencoder ----------------------------
            for i in range(args.niters_ae):
                if niter == len(train_data):
                    break  # end of epoch
                total_loss_ae, start_time = \
                    train_ae(train_data[niter], total_loss_ae, start_time, niter)
                niter += 1

            # train gan ----------------------------------
            for k in range(niter_gan):

                # train discriminator/critic
                for i in range(args.niters_gan_d):
                    # feed a seen sample within this epoch; good for early training
                    errD, errD_real, errD_fake = \
                        train_gan_d(train_data[random.randint(0, len(train_data)-1)])

                # train generator
                for i in range(args.niters_gan_g):
                    errG = train_gan_g()



            """
                The codes here are for logging running status, not actually conduct the algorithm logic
            """
            niter_global += 1
            if niter_global % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
                    'Loss_D_fake: %.8f) Loss_G: %.8f'
                    % (epoch, args.epochs, niter, len(train_data),
                        errD.data[0], errD_real.data[0],
                        errD_fake.data[0], errG.data[0]))
                with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                    f.write('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
                            'Loss_D_fake: %.8f) Loss_G: %.8f\n'
                            % (epoch, args.epochs, niter, len(train_data),
                            errD.data[0], errD_real.data[0],
                            errD_fake.data[0], errG.data[0]))

                # exponentially decaying noise on autoencoder
                autoencoder.noise_radius = \
                    autoencoder.noise_radius*args.noise_anneal


        # embed_corpus(corpus, args.emsize)
        embed_afterLSTM(corpus, args.nhidden)
        # save_model()

        # shuffle between epochs
        train_data = batchify(corpus.train, args.batch_size, shuffle=True)

    ###############################################################################
    # visualization
    ###############################################################################

    # input membership(label) of each node
    membership_path = "../data/membership.txt"

    # use last iteration output embeddings as code
    embedding_path = "./output/example/embed_afterLSTM_{}.txt".format(args.epochs)
    # load embeddings into numpy array


    # viz_tsne(membership_path, embedding_path)
    viz(membership_path, embedding_path)
    return embedding_path