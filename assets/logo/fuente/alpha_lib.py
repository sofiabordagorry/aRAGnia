import zlib, struct
from alpha import decode, encode
def compose(pw, pb, po):
    w,h,cw,W = decode(pw); _,_,cb,B = decode(pb)
    out = bytearray(w*h*4)
    for i in range(w*h):
        p=i*cw; q=i*cb; o=i*4
        a = 765 - (W[p]-B[q]) - (W[p+1]-B[q+1]) - (W[p+2]-B[q+2])
        if a<=0: out[o+3]=0
        elif a>=765: out[o],out[o+1],out[o+2],out[o+3]=B[q],B[q+1],B[q+2],255
        else:
            f=765.0/a
            r=int(B[q]*f); g=int(B[q+1]*f); b=int(B[q+2]*f)
            out[o]=min(r,255); out[o+1]=min(g,255); out[o+2]=min(b,255); out[o+3]=(a+1)//3
    encode(po,w,h,out)

def crop(path, pad=24):
    import zlib, struct
    w,h,ch,px = decode(path)
    assert ch==4
    x0,y0,x1,y1 = w,h,-1,-1
    for y in range(h):
        base=y*w*4
        for x in range(w):
            if px[base+x*4+3]:
                if x<x0: x0=x
                if x>x1: x1=x
                if y<y0: y0=y
                if y>y1: y1=y
    x0=max(0,x0-pad); y0=max(0,y0-pad); x1=min(w-1,x1+pad); y1=min(h-1,y1+pad)
    nw,nh=x1-x0+1, y1-y0+1
    out=bytearray(nw*nh*4)
    for y in range(nh):
        s=((y+y0)*w+x0)*4
        out[y*nw*4:(y+1)*nw*4]=px[s:s+nw*4]
    encode(path,nw,nh,out)
    return nw,nh
