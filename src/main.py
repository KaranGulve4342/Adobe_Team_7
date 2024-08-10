import regularize as rg
import plot as p

colours = ['r', 'g', 'b', 'y']
pathXY=p.read_csv('..\problems\isolated.csv')
regularized_path_XYs = []
p.plot(pathXY,colours)

for path in pathXY:
    regularized_path = []
    for shape in path:
        regularized_shape = rg.standardize_shape(shape)
        regularized_path.append(regularized_shape)
    regularized_path_XYs.append(regularized_path)
p.plot(regularized_path_XYs, colours)