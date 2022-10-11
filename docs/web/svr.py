webroot='/content/'
fpg='''<html><body><center><h1>tw url:</h1><input></input><br><br><a href="/proc">Show Procs</a><br><br><a href="/masking">Masking Tool</a></center></body></html>'''


procpage1='<html><body><style>PC {display: flow-root; margin-left: 5%;}</style>'
procpage2='''</body><script>
var h=['''
procpage3='''];
function build()
{
var zetr=document.getElementsByTagName("PC");
var zetl=zetr.length;
var zet = new Array(zetl);
for(var i=0;i<zetl;i++)
{
	zet[i]=zetr[i];
}
for(var i=0;i<zetl;i++)
{
var pik=h[i];
if(pik>-1){zet[pik].appendChild(zet[i]);}
}}
build();
</script></html>'''



from flask import Flask, request,send_file
import base64
import subprocess
import psutil
import os
import numpy as np
import cv2

app = Flask(__name__)
closewin='<script>close();</script>'

def readhtml(fna):
	with open(fna,'rt') as f:
		kt=f.read()
	return kt



@app.route('/')
def dodl():
	return fpg


@app.route('/sf/<path:pa>')
def static_file(pa):
	fpa=pa
	if not os.path.isfile(fpa):
		fpa=webroot+fpa
	if os.path.isfile(fpa):
		with open(fpa,'rb') as f:
			return f.read()
	else:
		return 'No '+fpa, 404


@app.route('/sf2/<path:pa>')
def downloadFile(pa):
	fpa=webroot+pa
	if os.path.isfile(fpa):
		return send_file(fpa, as_attachment=True)
	else:
		return 'No '+fpa, 404
    


			

dfumsk='user_mask.png'		

@app.route('/dl',methods=['GET', 'POST'])
def dlmp4():
	inv=request.args.get('inv')
	urrl= request.args.get('url').replace(' ','+')
	with open(dfumsk,'wb') as f:
		f.write(base64.b64decode(urrl))
	mk= cv2.imread( dfumsk , cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0
	mk=mk.transpose(2,0,1)[3]
	if inv == '1':
		mk=1.0-mk
	np.save('user_mask.npy',mk.astype(np.uint8))
	mk=cv2.resize(mk*255.0, None, fx=8, fy=8, interpolation =cv2.INTER_NEAREST )
	cv2.imwrite(dfumsk, mk.astype(np.uint8))
	return ''
	

@app.route('/dir/<sig>')
def listfo(sig):
	fli=os.listdir(webroot+sig)
	fli.sort()
	return ('<br>'.join(fli))+'<h1>'+str(len(fli))+'</h1>'



@app.route('/masking')
def showpainter():
	zt=readhtml('web/curmsk.txt').splitlines()
	psg0=readhtml('web/psg0.htm')
	psg1=readhtml('web/psg1.htm')
	psg1=psg1.replace('https://localhost:8333','/sf').replace('https://localhost:8133','')
	return psg0+zt[0]+';\nvar dYh='+zt[1]+";\nvar imgfna='"+zt[2]+"';\n\n"+psg1
	


@app.route('/proc')
def showproc():
	pid2ord=dict()
	ele=[]
	paarr=[]
	pidz=psutil.pids()
	k=0
	for pid in pidz:
		pid2ord[pid]=k
		k+=1
		po=psutil.Process(pid)
		pidstr=str(pid)
		paarr.append(po.ppid())
		cmdl='accDenied'
		try:
			cmdl=str(po.cmdline())
		except:
			pass
		ele.append('<PC><a href="/pkill/'+pidstr+'" >[[['+pidstr+']]]</a> '+cmdl+'<br></PC>')
	k=0
	for pu in paarr:
		rzu=-1
		try:
			rzuk=pid2ord[paarr[k]]
			if rzuk != k:
				rzu=rzuk
		except:
			pass
		paarr[k]=str(rzu)
		k+=1

	return procpage1+'\n'.join(ele)+procpage2+','.join(paarr)+procpage3

@app.route('/pkill/<pis>')
def killproc(pis):
	psutil.Process(int(pis,10)).terminate()
	return '',404


if __name__ == '__main__':
	os.rename('web/svr.py','web/svr.py_one')
	app.run(host='0.0.0.0', port=8133)
