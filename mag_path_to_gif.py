import magnes.graphics
import sys
import os


# convert -delay 20 -loop 0 *.jpg myimage.gif

def pathtogif(filename, save=None):
    if save == None: save = os.getcwd() + 'path.gif'
    container = magnes.io.load(filename)
    system = container.extract_system()
    path = container["PATH"]
    state = system.field3D()
    files = ''
    for idx, image in enumerate(path):
        state.upload(image)
        fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xz', sliceN=int(image.shape[1] / 2))
        fig.savefig('{:06d}.png'.format(idx), bbox_inches='tight')
        files += '{:06d}.png '.format(idx)
    os.system('convert -delay 20 ' + files)  # +save)
    os.system('rm ' + files)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pathtogif(sys.argv[1], sys.argv[2])
    else:
        pathtogif('mat/05/cone_path/cone_path_optimized.npz')  # sys.argv[1])
