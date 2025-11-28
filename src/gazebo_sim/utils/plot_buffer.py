import matplotlib.pyplot as plt ;
import numpy as np ;
import sys ;


(data,) = np.load(sys.argv[1]).values() ;

print(data.shape) ;

f,ax = plt.subplots(6,10) ;
f.set_figwidth(10)
f.set_figheight(6)

for (_ax,d) in zip(ax.ravel(),data):
  _ax.imshow(d) ;
  _ax.set_xticklabels([])
  _ax.set_yticklabels([])
  _ax.set_axis_off() ;
  _ax.set_aspect("equal") ;

#plt.gca().set_aspect("auto")
plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.01, right=0.99,top=0.99,bottom=0.01)
#plt.tight_layout()

plt.savefig(sys.argv[1].replace(".npz",".png")) ;


