import argparse

from train_6 import train
import wandb

NUM_MASSES = 4

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default= None, type=str, help='Name of the project on wandb')
    parser.add_argument('--env', default = "CustomHopper-source-v0", type = str, help='Name of the environment')
    parser.add_argument('--algo', default = "PPO", type = str, help='Name of the algorithm')
    parser.add_argument('--timesteps', default = 500000, type = int, help='Number of timesteps')
    parser.add_argument('--model_destination_path', default = 'models/DR', type = str, help='Destination path of the trained model')
    parser.add_argument('--save', default=False, action = 'store_true', help='call --save if you dont want to save the final model')
    parser.add_argument(
        '--model_config',
        default="{'policy' : 'MlpPolicy', 'verbose' : 1 ,'learning_rate': 3e-4, 'batch_size': 64, 'n_steps': 2048, 'n_epochs': 10, 'gamma' : 0.99, 'gae_lambda' : 0.99, 'clip_range': 0.2, 'ent_coef' : 0.2, 'tensorboard_log' : './ppo_hopper_tensorboard/'}",
        type=str,
        help="Hyperparameters dictionary for the model, it's important when throw the program from terminal to follow the default sintax"
    )
    parser.add_argument('--multipliers', default = '[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]', type = str, help='list of ranges to try, a range of 0.05 is something that allow the program to vary a specific mass from -5% to +5%')
    parser.add_argument('--mode', default = 'Simultaneous', type = str, help='Every multiplier will be tried simultaneously ')
    return parser.parse_args()

args = parse_args()

def tuning():
    multipliers = eval(args.multipliers) #list of bounds


    if args.mode == 'Simultaneous':

        for i, mul in enumerate(multipliers):
            DR_configuration = {
                'thigh': mul,
                'leg': mul,
                'foot': mul
            }

            run = None

            if args.project_name is not None:
                run = wandb.init(
                    project=args.project_name,
                    id=f"run_{i}",
                    config={'multiplier': mul},
                    sync_tensorboard=True
                )

            print(f"multiplier = {mul}")

            train(argparse.Namespace(
                project_name = args.project_name,
                env = args.env,
                algo = args.algo,
                timesteps = args.timesteps,
                model_destination_path = args.model_destination_path,
                save = args.save,
                model_config = args.model_config,
                env_config = str(DR_configuration),
            )) #Here NameSpace is something that allow you to throw a configuration like from terminal

            if run is not None:
                run.finish()


    '''
    here can be implemented a non simultaneous mode. It could consists of a configuration of different multipliers
    for different masses. For example, in this case we want to tune three different masses:
    first you randomize only the tight mass (and the other are not subject to DR), and you randomize with
    all the values in args.multipliers and see witch one performs better.
    Once you have found the best bounds for the tight, you do the same for other masses. When you have 
    found all the better multipliers for all the masses you can finally run a unique configuration with
    each mass linked to it's own value of multiplier
    '''

def main():
    tuning()


if __name__ == '__main__':
    main()