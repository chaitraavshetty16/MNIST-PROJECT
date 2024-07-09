from encoding import Spiking
from genetic import init_start_training
from output import classification

if __name__ == '__main__':
    encoding = Spiking()
    evolvedModels = init_start_training()
    classification(evolvedModels[0])


