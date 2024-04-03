from accelerate import Accelerator, DistributedDataParallelKwargs
from src.utils import load_ruamel
import src.models.utils as ut
import joblib
from src.optimization import CompleteOptunaOptimizer

def get_accelerator() -> Accelerator:
    kwargs = [
        DistributedDataParallelKwargs(find_unused_parameters=True),
    ]
    return Accelerator(kwargs_handlers=kwargs)


def main():
    params_file = ut.parse_args().params_path
    params = load_ruamel(params_file)
    accelerator = get_accelerator()
    optimizer = CompleteOptunaOptimizer(params=params, accelerator=accelerator)
    optimizer.optimize()
    best_trial = optimizer.get_best_trial()
    optimizer.remove_tmp()
    accelerator.print(best_trial)
    if accelerator.is_main_process:
        joblib.dump(best_trial, f'{params.get("out_dir")}/best_trial.pkl')


if __name__ == "__main__":
    main()
