import engine
import numpy as np
import utility


if __name__ == '__main__':
    print('Experiment Running.')
    args = utility.load_params(jsonFile='config.json')
    print(args)
    runner = engine.Engine(args)
    if args['generate']:
        x = input('Input a word:')
        runner.load_model()
        if args['data']['use_mini']:
            runner.generate_mini(x)
        else:
            runner.generate_big(x)
    else:
        if args['train']:
            if args['continue']:
                runner.load_model()
            for i in range(args['num_epochs']):
                runner.train()
                runner.save_model()
            runner.test()
            runner.plot()
        else:
            runner.load_model()
            runner.test()
            runner.plot()