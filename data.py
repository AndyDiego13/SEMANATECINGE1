#data.py
#andydiego

filepath = "./data/iris.data"


with open(filepath, 'r') as fp:
    data = fp.read()

data_lines = data.split('/n')

#data_final = []
#for line in data_lines:
#    data_final.append(line.split())

#LA FUNCIÓN DE ABAJO ES MEJOR QUE PONERLA ARRIBA

data_final = [f.split(',') for f in data_lines]

print(data_final)