
import utils
import configurations
import tensorflow as tf
from Networks.utils import make_weight_sharing













def main():
    args = configurations.get_args()
    network = utils.get_network(args.nntype)





if __name__ == "__main__":
    main()
