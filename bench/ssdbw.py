import os, sys, time, random, threading
path=sys.argv[1]; nthreads=int(sys.argv[2]); chunk=9954304; n=int(sys.argv[3])  # 9.49 MiB expert-sized reads
fd=os.open(path, os.O_RDONLY)
try: import fcntl; fcntl.fcntl(fd, 48, 1)  # F_NOCACHE=48 on macOS
except Exception as e: print("nocache:", e)
size=os.path.getsize(path); lo=int(sys.argv[4]) if len(sys.argv)>4 else int(size*0.05); hi=int(sys.argv[5]) if len(sys.argv)>5 else int(size*0.95)
offs=[random.randrange(lo, hi-chunk)//4096*4096 for _ in range(n)]
done=[0]*nthreads
def w(t):
    for i in range(t, n, nthreads):
        b=os.pread(fd, chunk, offs[i]); done[t]+=len(b)
t0=time.time(); th=[threading.Thread(target=w,args=(t,)) for t in range(nthreads)]
[x.start() for x in th]; [x.join() for x in th]; dt=time.time()-t0
print(f"threads={nthreads} reads={n} x {chunk/2**20:.1f} MiB: {sum(done)/dt/1e9:.2f} GB/s, {dt/n*1e3:.2f} ms/read amortized")
