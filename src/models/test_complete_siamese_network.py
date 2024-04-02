from train_complete_siamese_network import CompleteOptunaOptimizer, get_accelerator
import src.models.utils as ut
from src.utils import load_ruamel


def main():
    params_file = ut.parse_args().params_path
    params = load_ruamel(params_file)
    accelerator = get_accelerator()
    optimizer = CompleteOptunaOptimizer(params=params, accelerator=accelerator)
    optimizer.test()


if __name__ == "__main__":
    main()
