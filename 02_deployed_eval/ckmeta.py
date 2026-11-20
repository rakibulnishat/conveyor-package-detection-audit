import sys, zipfile, pickle, io
"""Print the metadata stored in an Ultralytics checkpoint and save its per-epoch training log.
usage: python experiments/02_deployed_eval/ckmeta.py apps/desktop/best.pt experiments/results/deployed_eval
Reads the pickle without importing Ultralytics classes, so it works on any machine."""
p=sys.argv[1]; out=sys.argv[2] if len(sys.argv)>2 else "."; z=zipfile.ZipFile(p)
names=z.namelist(); pk=[n for n in names if n.endswith('data.pkl')][0]
class Dummy:
    def __init__(self,*a,**k): self.a=a
    def __setstate__(self,s): self.s=s
    def __call__(self,*a,**k): return Dummy()
class U(pickle.Unpickler):
    def find_class(self,mod,name):
        if mod in('collections',) : return super().find_class(mod,name)
        if mod=='builtins': return super().find_class(mod,name)
        if mod.startswith('datetime'): return super().find_class(mod,name)
        return Dummy
    def persistent_load(self,pid): return ('storage',)
obj=U(io.BytesIO(z.read(pk))).load()
print(type(obj), list(obj.keys()) if isinstance(obj,dict) else '')
for k in ('date','version','license','docs','epoch','best_fitness','git'):
    print(k,'=',obj.get(k))
ta=obj.get('train_args') or {}
for k in ('data','model','epochs','imgsz','batch','seed','optimizer','lr0','warmup_bias_lr','mosaic','name','save_dir','tracker'):
    print('  train_args.',k,'=',ta.get(k))
print('train_metrics =',obj.get('train_metrics'))
tr=obj.get('train_results')
if isinstance(tr,dict):
    print('train_results keys:',list(tr.keys())); 
    import json; json.dump(tr,open(out+'/train_results.json','w'))
    import csv
    keys=list(tr.keys()); n=len(tr[keys[0]])
    with open(out+'/train_results.csv','w',newline='') as fh:
        w=csv.writer(fh); w.writerow(keys); [w.writerow([tr[k][i] for k in keys]) for i in range(n)]
    print('n epochs logged:', len(next(iter(tr.values()))))
