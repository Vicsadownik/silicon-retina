import argparse,sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx							
from ortools.sat.python import cp_model

'''
Conflict Resolution Algorithm in a Tangled Directed graphs or “two of every kind”.
Ten royal grooms want to choose brides from ten kingdoms as their wives, 
but at the same time the royal sympathies are ambiguous (everyone likes not one, but several princesses). 
How to unequivocally satisfy their expectations - so that each prince has a special princess 
and so that no one fights.
Fast and concise implementation of Constraint programming (CP) technology based on the free Google OR-Tools software.

'''
#-----------------------------------------------------------------------
'''
	plot demo
'''
def implotn(images,fig=7055, title=None, cmap='gray',label=None):
	fig = plt.figure( fig ); plt.clf(); plt.axis('off')
	if not title is None: plt.title(title)  
	n = len(images)
	grid = plt.GridSpec(1,n)
	axs = []
	for i in range(n):
		ax = fig.add_subplot(grid[0,i]) 
		ax.axis('off')
		ax.imshow(images[i],cmap=cmap)
		axs.append( ax )
	if not label is None: 
		plt.legend(prop={'size': 10},loc='upper left',bbox_to_anchor=(1,1),handles=label)
	return axs	

#-----------------------------------------------------------------------

'''
	Random graph (Tangled Directed) maker
	n - number of base nodes (total nodes = n * 2)
	m - max number of edges from node
	vi = {None, number}, if vi!=None then plot-demo
'''
def makerandomgraph(n=10,m=5,vi=None):
	G = nx.DiGraph()								# new dir graph							
	en = np.random.random_integers(1,m,n)						# number of edges from each base nodes кол-во ребер из каждой вершины от 1 до m
	ee = [ np.unique(np.random.random_integers(n,2*n-1,j)).tolist() for j in en ] 	# candidates nodes for each base nodes
	edges = [ (i,j) for i,e in enumerate(ee) for j in e ]				# all edges 
	G.add_edges_from(edges)								# add edges to graph
	if not vi is None:								# if demo-mode
		color_map = [ 'orange' if node < n else 'skyblue' for node in G ]
		zero = np.ones((100,100,3))
		ax = implotn([zero],title='Random confusing-directed-graph '+str([n,m]),fig=vi)
		pos = nx.spring_layout(G)
		o = zero.shape[0]/2
		for p in pos:  pos[p]+=o
		nx.draw_networkx(G,pos,ax=ax[0],node_color=color_map, edge_color='skyblue', with_labels=True)
	return G


'''
	Class for multisolutions 
'''
class VarArraySolutionCollector(cp_model.CpSolverSolutionCallback):
	def __init__(self, variables, limit=np.inf):
		self.solutionlimit = limit
		cp_model.CpSolverSolutionCallback.__init__(self)
		self.__variables = variables
		self.solution_list = []
	def on_solution_callback(self):
		print(':',end='',flush=True)
		if len(self.solution_list)>=self.solutionlimit:
			self.StopSearch()
		else:
			self.solution_list.append([self.Value(v) for v in self.__variables])	

#-----------------------------------------------------------------------

'''
	It takes all edges (all pairs of vertices) from the graph and forms rules from them as hashes (who refers to whom)
	kind of [[0, 1, 2, 3, 4], [5, 6, 7], [8], [9, 10], ..., [37]] - as many vertices as the graph + backward formula
	where the numbers are the indices of the edges of the graph.
	Returns an array of rules and an array of all edges of the graph
'''
def makerules(G):
	edges = np.array(list(G.edges()))		# graph edges
	nodes1,nodes2 = edges[:,0],edges[:,1]		# pairs nodes (base and candidates)
	a1 = {}		# direct hash: which cadidates each base link refers to 
	a2 = {}		# reverse hash: which base references each candidate 
	for i in range(len(nodes1)): 
		e1,e2 = nodes1[i],nodes2[i]
		if e1 in a1: a1[e1].append(e2)
		else: a1[e1]=[e2]
		if e2 in a2: a2[e2].append(e1)
		else: a2[e2]=[e1]
	rules = []			# rules accumulator
	k = 0				# counter for direct 
	for key1 in a1:								
		o = []									
		for l2 in a1[key1]:				
			o.append( k )
			k += 1
		rules.append( o )
	r0 = edges.tolist()
	for key2 in a2:			# revers
		o = []
		for l1 in a2[key2]:
			k = r0.index([l1,key2])	# counter for reverse 
			o.append( k )	
		rules.append( o )
	return edges,rules
	
#-----------------------------------------------------------------------

'''
	Make and run CP-SAT Solver for Tangled Directed graphs
'''
def extractalldigraph(G,limit=20,vi=None):
		
	R,O = makerules(G)							# get all edges and rules to create variables and constraints
	
	model = cp_model.CpModel() 						# declares the CP-SAT model
	X = [ model.NewIntVar(0,1, 'x'+str(i) ) for i in range(len(R)) ]	# creates the variables for the problem
	_= [ model.Add( np.sum([ X[i] for i in o ]) == 1 ) for o in O ]		# creates the constraint 
	solver = cp_model.CpSolver()						# calls the solver
	# solver.parameters.max_time_in_seconds = 12.0				# limit by processing time
																		# limit by number of solutions
	solution_collector = VarArraySolutionCollector(X,limit=limit)		# to find all feasible solutions		
	solver.SearchForAllSolutions(model, solution_collector)			# and place their to collector
	assert solution_collector.solutionlimit == limit
	
	rr = solution_collector.solution_list					# get all feasible solutions
	
	if len(rr)==0:		# NO feasible solution
		if vi: print('CP-SAT solver === NO feasible solution')
		return None
	
	if not vi is None: # if demo-mode
		
		r0 = rr[0]					# select [0] solution
		r = [ [ r0[i] for i in o ] for o in O ]		# by rules
		
		print('CP-SAT solver === OK, total',len(rr),'feasible solutions, limit:',limit) 
		
		n = len(G.nodes())/2					# total base nodes
		color_edges = ['skyblue']*len(R)			# set colors for edges
			
		for i,o in enumerate(O):				# set RED-color for nonzero (selected) edges
			j = o[np.where(np.array(r[i])>0)[0][0]]
			color_edges[j] = 'red' 
			
		color_map = [ 'red' if node < n else 'skyblue' for node in G ]	#  set colors for nodes
		
		zero = np.ones((100,100,3))				# zero image for demo
		
		# labels for edges
		h = [ mpatches.Patch(label=str(i)+' to '+str(R[o][:,1].tolist())) for i,o in enumerate(O[:int(n)]) ]
		
		# zero-image and titles demo 
		ax = implotn([zero],title='CP-SAT-demo solution[0] for nodes: '+str(int(n))+'+'+str(int(n))+', edges: '+str(len(R))+'\ntotal feasible solutions: '+str(len(rr)),fig=vi, label=h)
		
		# placement base and candidates nodes on zero-image
		pp = {} 
		shape = zero.shape
		stepx = (shape[0]-2) / n
		for node in G:
			if node < n:
				pp[node] = [1,1+node*stepx] 
			else:
				pp[node] = [shape[1]-2,1+(node-n)*stepx] 
		# graph demo 	
		nx.draw_networkx(G,pp,ax=ax[0],node_color=color_map, edge_color=color_edges, with_labels=True)
	
	return rr

#-----------------------------------------------------------------------

if sys.flags.interactive: 
		mpl.use('TkAgg')
		plt.ion()			 	

ap = argparse.ArgumentParser()

ap.add_argument( "-a", "--nattempts",   required = False,  type = int, help = "max number of attempts to get a feasible solutions",   default = 200 )	
ap.add_argument( "-n", "--nnodes",   required = False,  type = int, help = "number of based nodes",   default = 25 )
ap.add_argument( "-m", "--medges",   required = False,  type = int, help = "max number of edges from one node to others",   default = 15 )
ap.add_argument( "-l", "--msolutions",   required = False,  type = int, help = "max number of solutions (stopSolver limit)",   default = 20 )	
args = vars(ap.parse_args())
	
a,n,m,l = args[ "nattempts" ],args[ "nnodes" ],args[ "medges" ],args[ "msolutions" ]
	
spar = "[nattempts:"+str(a)+"][nnodes:"+str(n)+"][medges:"+str(m)+'][msolutions:'+str(l)+']'
	
print("\n-----------------------------------------------------------------------------------------")	
print("'Two of every kind' v1.0 07/06/20 !","t" if sys.flags.interactive else "m",spar)
print("   run: python3 -i sr3s.py -a100 -n12 -m5")
print("-----------------------------------------------------------------------------------------\n")


for i in range(a):
	G = makerandomgraph(n=n,m=m,vi=None); 
	ret = extractalldigraph(G,limit=l,vi=1)
	if not ret is None: 
		break
plt.show()		
		
		
