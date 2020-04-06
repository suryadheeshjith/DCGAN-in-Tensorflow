import argparse

from gan import gan_run
from models import discriminator, generator

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-lr',"--learning_rate", dest="learning_rate", type=float, default=1e-3, help="Learning Rate in Adam. Default=1e-3")
    parser.add_argument('-b',"--beta1", dest="beta_1", type=float, default=0.99, help="Beta in Adam. Default=0.5")
    parser.add_argument('-e',"--epochs", dest="epochs", type=int, default=10, help="Epochs. Default=10")
    parser.add_argument('-bs',"--batch_size", dest="batch_size", type=int, default=128, help="Batch Size. Default =128")
    parser.add_argument('-ns',"--noise_size", dest="noise_size", type=int, default=96, help="Noise Size. Default=96")
    parser.add_argument('-p',"--print_every", dest="print_every", type=int, default=20, help="Print at how many steps. Default=20")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Arguments used
    args = parse_args()
    print("\n\nArguments Used : "+str(args)+"\n\n\n")

    learning_rate = args.learning_rate
    beta_1 = args.beta_1
    epochs = args.epochs
    batch_size = args.batch_size
    noise_size = args.noise_size
    print_every = args.print_every




    D = discriminator()
    G = generator()
    gan_run(D, G,epochs,learning_rate,beta_1, print_every, batch_size,noise_size)
