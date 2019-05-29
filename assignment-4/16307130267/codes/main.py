import engine
import fnlp_engine
import utility

if __name__ == '__main__':
	args = utility.load_params(jsonFile='config.json')
	if args['data']['dataset'] == 'zh':
		runner = engine.Engine(args)
	elif args['data']['dataset'] == 'en':
		runner = fnlp_engine.Engine(args)
	else:
		print('Invalid dataset!')
		exit()

	if args['data']['dataset'] == 'zh' and args['predict']:
		runner.load_model()
		#x = raw_input('Input a sentence: ')
		x = input('Input a sentence: ')
		runner.predict(x)
	else:
		if args['train']:
			if args['continue']:
				runner.load_model()
			runner.train()
			#runner.save_model()
			runner.test()
		else:
			runner.load_model()
			runner.test()