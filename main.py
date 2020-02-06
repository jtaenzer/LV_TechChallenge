from data_interpreter import DataInterpreter

data_interp = DataInterpreter()
input_data, output = data_interp.sequences[:, :-1], data_interp.sequences[:, -1]
