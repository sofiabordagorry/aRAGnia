import zlib, struct, time

def decode(path):
    d = open(path,'rb').read(); i = 8; idat = b''
    while i < len(d):
        ln = struct.unpack(">I", d[i:i+4])[0]; typ = d[i+4:i+8]
        if typ == b'IHDR': w,h,bd,ct = struct.unpack(">IIBB", d[i+8:i+18])
        elif typ == b'IDAT': idat += d[i+8:i+8+ln]
        i += 12 + ln
    assert bd == 8 and ct in (2,6), (bd,ct)
    ch = 4 if ct == 6 else 3
    raw = zlib.decompress(idat); stride = w*ch
    out = bytearray(h*stride); prev = bytearray(stride); pos = 0
    for y in range(h):
        f = raw[pos]; pos += 1
        cur = bytearray(raw[pos:pos+stride]); pos += stride
        if f == 1:
            for x in range(ch, stride): cur[x] = (cur[x] + cur[x-ch]) & 255
        elif f == 2:
            for x in range(stride): cur[x] = (cur[x] + prev[x]) & 255
        elif f == 3:
            for x in range(stride):
                a = cur[x-ch] if x >= ch else 0
                cur[x] = (cur[x] + ((a + prev[x]) >> 1)) & 255
        elif f == 4:
            for x in range(stride):
                a = cur[x-ch] if x >= ch else 0
                b = prev[x]; c = prev[x-ch] if x >= ch else 0
                p = a + b - c; pa = abs(p-a); pb = abs(p-b); pc = abs(p-c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[x] = (cur[x] + pr) & 255
        out[y*stride:(y+1)*stride] = cur; prev = cur
    return w, h, ch, out

def encode(path, w, h, rgba):
    raw = bytearray()
    stride = w*4
    for y in range(h):
        raw.append(0); raw += rgba[y*stride:(y+1)*stride]
    comp = zlib.compress(bytes(raw), 9)
    def chunk(t, data):
        c = t + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) \
          + chunk(b'IDAT', comp) + chunk(b'IEND', b'')
    open(path,'wb').write(png)

