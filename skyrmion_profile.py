#!/usr/bin/env python
import numpy as np
import sys
import os
os.environ['MAGNES_BACKEND'] = 'numpy'
import os.path
import magnes
import magnes.graphics
import magnes.utils
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline
import scipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import scipy.signal as signal

def plot_state(system,s,directory,file,show_extras=False):
	directory=Path(directory)
	state = system.field3D()
	state.upload(s)
	fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xz', sliceN=int(system.size[1] / 2))
	fig.savefig(directory.joinpath(file+'_xz.pdf'), bbox_inches='tight')
	plt.close('all')
	fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='yz', sliceN=int(system.size[0] / 2))
	fig.savefig(directory.joinpath(file+'_yz.pdf'), bbox_inches='tight')
	plt.close('all')
	fig, _, _ = magnes.graphics.plot_field3D(system, state, slice2D='xy', sliceN=int(system.size[2] / 2))
	fig.savefig(directory.joinpath(file+'_xy.pdf'), bbox_inches='tight')
	plt.close('all')
	if show_extras:
		plt.plot(s[int(s.shape[0]/2),int(s.shape[1]/2),:,0,2])
		plt.xlabel(r'$z$')
		plt.ylabel(r'$m_z$')
		plt.savefig(directory.joinpath(file + '_z.pdf'), bbox_inches='tight')
		plt.close('all')
		plt.plot(s[int(s.shape[0] / 2), int(s.shape[1] / 2), :, 0, 1])
		plt.ylabel(r'$m_x$')
		plt.xlabel(r'$z$')
		plt.savefig(directory.joinpath(file + '_x.pdf'), bbox_inches='tight')
		plt.close('all')

def plot_npz(file,show_extras=False):
	try:
		file=Path(file)
		print(f'{file = }')
		container = magnes.io.load(str(file))
		system = container.extract_system()
		if 'STATE' in container:
			s = np.array(container["STATE"])
			plot_state(system=system, s=s, directory=file.parent, file=file.stem,show_extras=show_extras)
		else:
			print(f'{container["PATH"].shape = }')
			print(f'{container["PATH"].shape[0] = }')
			if container['PATH'].shape[0] == 1:
				s = list(container["PATH"])[0]
				plot_state(system=system, s=s, directory=file.parent, file=file.stem,show_extras=show_extras)
			else:
				for idx, s in enumerate(container['PATH']):
					plot_state(system=system, s=s, directory=file.parent.joinpath(file.stem), file=file.stem + '_{}'.format(idx),show_extras=show_extras)

	except:
		print('failed')


def skyrmion_profile(file,criteria=1.9,show=False):
#	if isinstance(file,np.ndarray):
#		s=file
#	else:
	file = Path(file)
	container = magnes.io.load(str(file))
	if 'STATE' in container:
		s = np.array(container["STATE"])
	else:
		s = list(container["PATH"])[0]
	size=[]
	#s=s[:,:97,:,:]
	x = np.array(range(s.shape[0]), dtype='float') - s.shape[0] / 2
	y = np.array(range(s.shape[1]), dtype='float') - s.shape[1] / 2
	z= np.array(range(s.shape[2]), dtype='float')
	x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
	angle_vector=[]
	shape=[]
	cone=np.squeeze(s[0,0,:,0,:])
	state_diff=np.linalg.norm(np.squeeze(s)-cone,axis=3)
	skyrmion_mask=state_diff>criteria
	#print(f'{skyrmion_mask.shape =  },{skyrmion_mask.sum() = }')
	if skyrmion_mask.sum()!=0:
		x_centre = np.nan
		y_centre = np.nan
		print(f'{skyrmion_mask[:,:,0].sum() = }')
		print(f'{skyrmion_mask.sum() = }')
		if skyrmion_mask[:,:,0].sum()>5:
			x_centre = np.mean(x_grid[skyrmion_mask[:,:,0]])
			y_centre = np.mean(y_grid[skyrmion_mask[:,:,0]])
		elif skyrmion_mask.sum()>40:
			x_2d_grid, y_2d_grid = np.meshgrid(x, y, indexing='ij')
			x_centre = np.mean(x_2d_grid[skyrmion_mask.sum(axis=2) > 0])
			y_centre = np.mean(y_2d_grid[skyrmion_mask.sum(axis=2) > 0])
		if not (np.isnan(x_centre)) and not (np.isnan(y_centre)):
			x = x - x_centre
			y = y - y_centre
		x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
		# print(f'{x_grid.shape = },{y_grid.shape = },{z_grid.shape = }')
		x_2d_grid, y_2d_grid = np.meshgrid(x, y, indexing='ij')
		if show:
			try:
				r_grid=np.sqrt(x_grid*x_grid+y_grid*y_grid)
				theta_grid=np.arctan2(y_grid, x_grid)
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				ax.scatter(x_grid[skyrmion_mask], y_grid[skyrmion_mask], z_grid[skyrmion_mask],c=theta_grid[skyrmion_mask]/theta_grid[skyrmion_mask].max())
				ax.set_xlabel('X')
				ax.set_ylabel('Y')
				ax.set_zlabel('Z')
				plt.show()
			except:
				plt.close('all')
		coord = []
		for i in range(s.shape[2]):
			is_sk=skyrmion_mask[:,:,i]
			r_2d_grid=np.sqrt(x_2d_grid*x_2d_grid+y_2d_grid*y_2d_grid)
			r_2d_grid[np.invert(is_sk)]=np.nan
			r_max=np.nanmax(r_2d_grid)
			r_min = np.nanmin(r_2d_grid)
			#print(f'{r_min = },{r_max = }')
			if r_min<=1:
				r_max_arg=np.unravel_index(np.nanargmax(r_2d_grid),r_2d_grid.shape)
				theta_grid = np.arctan2(y_2d_grid, x_2d_grid)
				theta_max=np.arctan2(y_2d_grid[r_max_arg],x_2d_grid[r_max_arg])

				fr=0.1
				if theta_max>fr:
					theta_mask=np.logical_and(theta_grid > theta_max- np.pi- fr, theta_grid < theta_max- np.pi+ fr )
				elif theta_max < -fr:
					theta_mask = np.logical_and(theta_grid > theta_max + np.pi - fr,
												theta_grid < theta_max + np.pi + fr)
				else:
					theta_mask = np.logical_and(theta_grid > theta_max + np.pi - fr,
												theta_grid < theta_max - np.pi + fr)
				#plt.plot(x_2d_grid[theta_mask],y_2d_grid[theta_mask],'.')
				#plt.xlim([-50,50])
				#plt.ylim([-50, 50])
				#plt.show()
				r_2d_grid[np.invert(theta_mask)]=np.nan
				r_min_=-np.nanmax(r_2d_grid)
				if np.isnan(r_min_):
					r_min=-r_min
				else:
					r_min = r_min_
		#		print(f'{i = },{r_min = },{r_max = }')
			coord.append([r_min,r_max])

		coord = np.array(coord)
		r_min = coord[:, 0]
		r_max = coord[:, 1]
		if np.all(np.invert(np.isnan(r_min))) and np.all(np.invert(np.isnan(r_max))):
			r_min=signal.wiener(r_min)
			r_max = signal.wiener(r_max)

		ax = plt.figure().gca()
		z_grid=np.array(range(len(r_min)))
		ax.plot(r_min[np.invert(np.isnan(r_min))],z_grid[np.invert(np.isnan(r_min))], 'b.',label='$r_{min}$')
		ax.plot(r_max[np.invert(np.isnan(r_max))],z_grid[np.invert(np.isnan(r_max))], 'r.',label='$r_{max}$')
		ax.set_ylabel('z')
		ax.set_xlabel('r')
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		plt.legend()
		plt.savefig(str(file.parent.joinpath(file.stem + '_r.pdf')))
		if show: plt.show()
		plt.close('all')

		return r_min,r_max, skyrmion_mask.sum()
	else:
		return np.nan, np.nan

# directions

#	for i in range(s.shape[2]):
#		z = s[:, :, i, 0, 1]
#		cone=s[0,0,i,0,:]
#		cone_sq=np.squeeze(np.linalg.norm(s[:,:,i,0,:]-cone,axis=2))
#		cone_mask=cone_sq>criteria
#		direction = s[0, 0, i, 0, :2]
#		direction = direction / np.linalg.norm(direction)
#		tan = direction[1] / direction[0]
#		angle = np.arctan2(direction[1], direction[0])/np.pi*180
#		perp_angle = angle - 90
#		perp_tan = np.tan(perp_angle/180*np.pi)

#		print(cone_mask.sum())
#		x_centre=np.mean(x_grid[cone_mask])
#		y_centre = np.mean(y_grid[cone_mask])
#		top_angle=np.arctan2(y_centre,x_centre)* 180 / np.pi
#		top_tan=y_centre/x_centre
#		angle_vector.append([i, angle, perp_angle, top_angle])
#		size.append([i,cone_mask.sum()])
#		interp = scipy.interpolate.interp2d(x_grid, y_grid, cone_sq, kind='linear')
#		if np.abs(tan) < 1:
#			line_x = x
#			line_y = tan * x
#		else:
#			line_y = y
#			line_x = y / tan
#		if np.abs(perp_tan) < 1:
#			perp_line_x = x
#			perp_line_y = perp_tan * x
#		else:
#			perp_line_y = y
#			perp_line_x = y / perp_tan
#		if np.abs(top_tan) < 1:
#			top_line_x = x
#			top_line_y = top_tan * x
#		else:
#			top_line_y = y
#			top_line_x = y / top_tan
#
#		top_line_z = np.array([interp(a, b) for a, b in zip(top_line_x, top_line_y)])
#		#plt.plot(top_line_x, top_line_z)
#		#plt.show()
#		top_d_arg = [np.array(range(len(top_line_z)))[(top_line_z>criteria).reshape(-1)].min(),np.array(range(len(top_line_z)))[(top_line_z>criteria).reshape(-1)].max()]
#		top_d_coord = np.array([[top_line_x[top_d_arg[0]], top_line_y[top_d_arg[0]]],
#		top_d_r=[np.linalg.norm(top_d_coord[0,:]),np.linalg.norm(top_d_coord[1,:])]
#		#plt.plot(top_line_x,top_line_z)
#		#plt.plot(top_d_coord[0,0],criteria,'.')
#		#plt.plot(top_d_coord[1,0], criteria, '.')
#		#plt.show()
#
#		shape.append(top_d_r)
#		plt.contourf(x_grid, y_grid, cone_sq)#z
#		plt.plot(x_centre, y_centre, 'k.')
#		plt.plot(line_x, line_y, 'b')
#		plt.plot(perp_line_x, perp_line_y, 'r')
#		plt.plot(top_line_x, top_line_y, 'g')
#		plt.plot(top_line_x[top_d_arg[0]], top_line_y[top_d_arg[0]],'rx')
#		plt.plot(top_line_x[top_d_arg[1]], top_line_y[top_d_arg[1]], 'bx')
#		plt.colorbar()
#		plt.title('layer {}'.format(i))
#		if show: plt.show()
#		plt.close('all')
#	shape=np.array(shape)
#	plt.plot(shape[:,0],'r.')
#	plt.plot(shape[:, 1],'b.')
#	plt.savefig(str(file.parent.joinpath(file.stem + '_skplace.pdf')))
#	if show: plt.show()
#	plt.close('all')
#	print(size)
#	angle_vector = np.array(angle_vector)
#	size=np.array(size)
#	plt.plot(size[:,0],size[:,1],'.')
#	plt.xlabel('z')
#	plt.ylabel('skyrmion square')
#	plt.savefig(str(file.parent.joinpath(file.stem+'_sksquare.pdf')))
#	if show: plt.show()
#	plt.close('all')
#
#	plt.plot(angle_vector[:, 0], angle_vector[:, 1], 'r.', label='spiral angle')
#	plt.plot(angle_vector[:, 0], angle_vector[:, 2], 'g.', label='spiral angle + 90')
#	plt.plot(angle_vector[:, 0], angle_vector[:, 3], 'b.', label='skyrmion angle to z max')
#	plt.xlabel('z')
#	plt.ylabel('angle')
#	plt.legend()
#	plt.savefig(str(file.parent.joinpath(file.stem + '_angle_alt.pdf')))
#	if show: plt.show()
#	plt.close('all')

def skyrmion_profile_max(file,show=False):
	file = Path(file)
	container = magnes.io.load(str(file))
	if 'STATE' in container:
		s = np.array(container["STATE"])
	else:
		s = list(container["PATH"])[0]

	angle_vector=[]
	size=[]
	for i in range(s.shape[2]):
		z = s[:, :, i, 0, 1]
		#z=np.zeros(z.shape)
		#z[20,1]=1
		print(f'{np.unravel_index(z.argmax(), z.shape) = }')
		point = np.unravel_index(z.argmax(), z.shape)
		point_xy=np.array([point[0]-z.shape[0]/2,point[1]-z.shape[1]/2])
		print(f'{point = }\t{point_xy = }\t{z[point] = }')
		top_direction = point_xy / np.linalg.norm(point_xy)
		top_tan=top_direction[1]/top_direction[0]
		top_angle=np.arctan2(top_direction[1],top_direction[0])*180/np.pi
		direction = s[0,0,i,0,:2]
		direction=direction/np.linalg.norm(direction)
		tan=direction[1]/direction[0]
		angle=np.arctan2(direction[1],direction[0])/np.pi*180
		perp_angle=angle-90
		perp_tan=np.tan(perp_angle*np.pi/180)
		angle_vector.append([i,angle,perp_angle,top_angle])
		x=np.array(range(z.shape[0]),dtype='float')-z.shape[0]/2
		y=np.array(range(z.shape[1]),dtype='float')-z.shape[1]/2
		x_grid,y_grid=np.meshgrid(x,y)
		x_grid=x_grid.T
		y_grid=y_grid.T
		interp=scipy.interpolate.interp2d(x_grid, y_grid,z, kind='linear')
		if np.abs(tan)<1:
			line_x=x
			line_y=tan*x
		else:
			line_y = y
			line_x = y/tan
		if np.abs(perp_tan)<1:
			perp_line_x=x
			perp_line_y=perp_tan*x
		else:
			perp_line_y = y
			perp_line_x = y/perp_tan

		if np.abs(top_tan)<1:
			top_line_x=x
			top_line_y=top_tan*x
		else:
			top_line_y = y
			top_line_x = y/top_tan
		line_z = np.array([interp(a, b) for a, b in zip(line_x, line_y)])
		perp_line_z = np.array([interp(a, b) for a, b in zip(perp_line_x, perp_line_y)])
		top_line_z = np.array([interp(a, b) for a, b in zip(top_line_x, top_line_y)])
		#plt.contourf(x_grid,y_grid,interp(x,y))
		top_d_arg=[top_line_z.argmax(),top_line_z.argmin()]
		top_d_coord=np.array([[top_line_x[top_d_arg[0]],top_line_y[top_d_arg[0]]],[top_line_x[top_d_arg[1]],top_line_y[top_d_arg[1]]]])
		top_d=np.linalg.norm(top_d_coord[1]-top_d_coord[0])
		size.append([i,top_d/2])

		plt.contourf(x_grid, y_grid,z)
		plt.plot(point_xy[0],point_xy[1],'k.')
		plt.title('layer {}'.format(i))
		plt.plot(line_x,line_y,'b')
		plt.plot(perp_line_x, perp_line_y,'r')
		plt.plot(top_line_x, top_line_y, 'g')
		plt.gcf().gca().add_artist(plt.Circle([0,0],top_d/2,fill=False))
		plt.colorbar()
		plt.xlabel('x')
		plt.ylabel('y')
		plt.savefig(str(file.parent.joinpath(file.stem+'_pdf.pdf')))
		if show: plt.show()
		plt.close('all')

		plt.plot(line_x, line_z, 'b',label='line')
		plt.plot(perp_line_x,perp_line_z,'r',label='perp line')
		plt.plot(top_line_x, top_line_z, 'g', label='top line')
		plt.legend()
		plt.xlabel('x')
		plt.ylabel('z')
		plt.title('layer {}'.format(i))
		if show: plt.show()
		plt.close('all')
	angle_vector=np.array(angle_vector)
	size=np.array(size)
	#angle_vector[:, 1]=angle_vector[:, 1] / np.pi * 180
	#angle_vector[:, 2] = angle_vector[:, 2] / np.pi * 180
	#angle_vector[:, 3] = angle_vector[:, 3] / np.pi * 180
	#angle_vector[angle_vector[:, 1]>90, 1]=angle_vector[angle_vector[:, 1]>90, 1]-180
	#angle_vector[angle_vector[:, 2] > 90, 2] = angle_vector[angle_vector[:, 2] > 90, 2] - 180
	#angle_vector[angle_vector[:, 3] > 90, 3] = angle_vector[angle_vector[:, 3] > 90, 3] - 180
	plt.plot(angle_vector[:,0],angle_vector[:,1],'r.',label='spiral angle')
	plt.plot(angle_vector[:, 0], angle_vector[:, 2], 'g.', label='spiral angle + 90')
	plt.plot(angle_vector[:, 0], angle_vector[:, 3], 'b.', label='skyrmion angle to z max')
	plt.xlabel('z')
	plt.ylabel('angle')
	plt.legend()
	plt.savefig(str(file.parent.joinpath(file.stem+'_angle.pdf')))
	if show: plt.show()
	plt.close('all')
	plt.plot(size[:,0],size[:,1],'.')
	plt.xlabel('z')
	plt.ylabel('skyrmion radius')
	plt.savefig(str(file.parent.joinpath(file.stem+'_sksize.pdf')))
	if show: plt.show()
	plt.close('all')

if __name__ == "__main__":
	show=False
	file=Path(sys.argv[1])
	if file.suffix=='.npz':
		print(skyrmion_profile(file, show=show))
	elif file.is_dir():
		data=[]
		for f in file.iterdir():
			if f.suffix == '.npz':
				print(skyrmion_profile(f, show=show)[2])
	#skyrmion_profile_max(sys.argv[1],show=show)
