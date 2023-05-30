import copy
import os
import warnings
import re
import numpy as np
import lzf
import struct

HAS_SENSOR_MSGS = True
try:
    from sensor_msgs.msg import PointField
    import numpy_pc2  # needs sensor_msgs
except ImportError:
    HAS_SENSOR_MSGS = False


class point_cloud(object):
    def __init__(self, metadata, pc_data):
        self.metadata_keys = metadata.keys()
        self.__dict__.update(metadata)
        self.pc_data = pc_data
        self.required = ('version', 'fields', 'size', 'width', 'height', 'points', 'viewpoint', 'data')
        self._check_sanity()

    def _get_metadata(self):
        metadata = {}
        for k in self.metadata_keys:
            metadata[k] = copy.copy(getattr(self, k))
        return metadata

    def _check_sanity(self):
        # pdb.set_trace()
        md = self._get_metadata()
        assert (self._metadata_is_consistent(md))
        assert (len(self.pc_data) == self.points)
        assert (self.width * self.height == self.points)
        assert (len(self.fields) == len(self.count))
        assert (len(self.fields) == len(self.type))

    def _metadata_is_consistent(self, metadata):
        checks = []
        for f in self.required:
            if f not in metadata:
                print('%s required' % f)
        checks.append((lambda m: all([k in m for k in self.required]),
                       'missing field'))
        checks.append((lambda m: len(m['type']) == len(m['count']) ==
                                 len(m['fields']),
                       'length of type, count and fields must be equal'))
        checks.append((lambda m: m['height'] > 0,
                       'height must be greater than 0'))
        checks.append((lambda m: m['width'] > 0,
                       'width must be greater than 0'))
        checks.append((lambda m: m['points'] > 0,
                       'points must be greater than 0'))
        checks.append((lambda m: m['data'].lower() in ('ascii', 'binary',
                                                       'binary_compressed'),
                       'unknown data type:'
                       'should be ascii/binary/binary_compressed'))
        ok = True
        for check, msg in checks:
            if not check(metadata):
                print('error:', msg)
                ok = False
        return ok

    def save_pcd(self, fname, data_compression='binary_compressed'):
        def write_header(_metadata, rename_padding=False):
            """ given metadata as dictionary return a string header.
            """
            template = """\
VERSION {version}
FIELDS {fields}
SIZE {size}
TYPE {type}
COUNT {count}
WIDTH {width}
HEIGHT {height}
VIEWPOINT {viewpoint}
POINTS {points}
DATA {data}\n"""
            str_metadata = _metadata.copy()

            if not rename_padding:
                str_metadata['fields'] = ' '.join(_metadata['fields'])
            else:
                new_fields = []
                for f in _metadata['fields']:
                    if f == '_':
                        new_fields.append('padding')
                    else:
                        new_fields.append(f)
                str_metadata['fields'] = ' '.join(new_fields)
            str_metadata['size'] = ' '.join(list(map(str, _metadata['size'])))
            str_metadata['type'] = ' '.join(_metadata['type'])
            str_metadata['count'] = ' '.join(list(map(str, _metadata['count'])))
            str_metadata['width'] = str(_metadata['width'])
            str_metadata['height'] = str(_metadata['height'])
            str_metadata['viewpoint'] = ' '.join(list(map(str, _metadata['viewpoint'])))
            str_metadata['points'] = str(_metadata['points'])
            tmpl = template.format(**str_metadata)
            return tmpl

        def build_ascii_fmtstr(pc_):
            """ make a format string for printing to ascii, using fields
            %.8f minimum for rgb
            %.10f for more general use?
            """
            fmtstr = []
            for t, cnt in zip(pc_.type, pc_.count):
                if t == 'F':
                    fmtstr.extend(['%.10f'] * cnt)
                elif t == 'I':
                    fmtstr.extend(['%d'] * cnt)
                elif t == 'U':
                    fmtstr.extend(['%u'] * cnt)
                else:
                    raise ValueError("don't know about type %s" % t)
            return fmtstr

        with open(fname, 'w') as fileobj:
            metadata = self._get_metadata()
            if data_compression is not None:
                data_compression = data_compression.lower()
                assert (data_compression in ('ascii', 'binary', 'binary_compressed'))
                metadata['data'] = data_compression

            header = write_header(metadata)
            fileobj.write(header)
            if metadata['data'].lower() == 'ascii':
                fmtstr = build_ascii_fmtstr(self)
                np.savetxt(fileobj, self.pc_data, fmt=fmtstr)
            elif metadata['data'].lower() == 'binary':
                fileobj.write(self.pc_data.tostring('C'))
            elif metadata['data'].lower() == 'binary_compressed':
                uncompressed_lst = []
                ##-------Claude-------##
                # for fieldname in pc.pc_data.dtype.names:
                #     column = np.ascontiguousarray(pc.pc_data[fieldname]).tostring('C')
                #     uncompressed_lst.append(column)
                ##----------------##
                # a=pc.pc_data.transpose()
                column = np.ascontiguousarray(self.pc_data.transpose()).tostring('C')
                uncompressed_lst.append(column)
                ##----------------##
                # uncompressed = ''.join(uncompressed_lst)
                uncompressed = column
                uncompressed_size = len(uncompressed)
                # print("uncompressed_size = %r"%(uncompressed_size))
                buf = lzf.compress(uncompressed)
                if buf is None:
                    # compression didn't shrink the file
                    # TO-DO what do to do in this case when reading?
                    buf = uncompressed
                    compressed_size = uncompressed_size
                else:
                    compressed_size = len(buf)
                fmt = 'ii'
                # a=struct.pack(fmt, compressed_size, uncompressed_size)
                # c=str(a, encoding='ascii')
                # b=map(ord, struct.pack(fmt, compressed_size, uncompressed_size))
                fileobj.write(struct.pack(fmt, compressed_size, uncompressed_size))
                fileobj.write(buf)
            else:
                raise ValueError('unknown DATA type')

    @staticmethod
    def to_msg(pc_data_raw, header='velodyne'):
        if not HAS_SENSOR_MSGS:
            raise Exception('ROS sensor_msgs not found')

        return numpy_pc2.array_to_pointcloud2(pc_data_raw, header)

    @staticmethod
    def save_points_as_pcd(pc_np, filename, data_compression='binary_compressed'):
        assert (len(pc_np.shape) == 2 and pc_np.shape[
            -1] == 4), "Function save_points_as_pcd: input data's shape is not compatible"
        cnt = pc_np.shape[0]
        metadata = dict({'count': [1, 1, 1, 1],
                         'data': 'ascii',
                         'fields': ['x', 'y', 'z', 'intensity'],
                         'height': 1,
                         'points': cnt,
                         'size': [4, 4, 4, 4],
                         'type': ['F', 'F', 'F', 'F'],
                         'version': '0.7',
                         'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                         'width': cnt,
                         })
        pointcloud = point_cloud(metadata, pc_np)
        pointcloud.save_pcd(filename, data_compression)

    @staticmethod
    def load_pcd_from_path(fname, from_bag=False, include_ring = False):
        """ parse pointcloud coming from file
        """

        def parse_header(lines):
            metadata = {}
            for ln in lines:
                ln = ln.decode('ascii')
                if ln.startswith('#') or len(ln) < 2:
                    continue
                match = re.match('(\w+)\s+([\w\s\.]+)', ln)
                if not match:
                    print("warning: can't understand line: %s" % ln)
                    continue
                key, value = match.group(1).lower(), match.group(2)
                if key == 'version':
                    metadata[key] = value
                elif key in ('fields', 'type'):
                    metadata[key] = value.split()
                elif key in ('size', 'count'):
                    metadata[key] = list(map(int, value.split()))
                elif key in ('width', 'height', 'points'):
                    metadata[key] = int(value)
                elif key == 'viewpoint':
                    metadata[key] = list(map(float, value.split()))
                elif key == 'data':
                    metadata[key] = value.strip().lower()
                # TO-DO apparently count is not required?
            # add some reasonable defaults
            if 'count' not in metadata:
                metadata['count'] = [1] * len(metadata['fields'])
            if 'viewpoint' not in metadata:
                metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            if 'version' not in metadata:
                metadata['version'] = '.7'
            return metadata

        def _build_dtype(metadata_):
            fieldnames = []
            typenames = []
            numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                                       (np.dtype('float64'), ('F', 8)),
                                       (np.dtype('uint8'), ('U', 1)),
                                       (np.dtype('uint16'), ('U', 2)),
                                       (np.dtype('uint32'), ('U', 4)),
                                       (np.dtype('uint64'), ('U', 8)),
                                       (np.dtype('int8'), ('I', 1)),
                                       (np.dtype('int16'), ('I', 2)),
                                       (np.dtype('int32'), ('I', 4)),
                                       (np.dtype('int64'), ('I', 8))]
            pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)
            flag = 0  # TODO:
            for f, c, t, s in zip(metadata_['fields'],
                                  metadata_['count'],
                                  metadata_['type'],
                                  metadata_['size']):
                np_type = pcd_type_to_numpy_type[(t, s)]
                if c == 1:
                    fieldnames.append(f)
                    typenames.append(np_type)
                else:

                    fieldnames.extend([str(flag) + '%s_%04d' % (f, i) for i in range(c)])
                    typenames.extend([np_type] * c)
                    flag += 1
            dtype = np.dtype(list(zip(fieldnames, typenames)))
            return dtype

        def parse_binary_pc_data(f, dtype, metadata):
            rowstep = metadata['points'] * dtype.itemsize
            buf = f.read(rowstep)
            return np.fromstring(buf, dtype=dtype)

        def parse_binary_compressed_pc_data(f, dtype, metadata):
            # compressed size of data (uint32)
            # uncompressed size of data (uint32)
            # compressed data
            # junk
            fmt = 'II'
            compressed_size, uncompressed_size = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
            compressed_data = f.read(compressed_size)
            # (compressed > uncompressed)
            # should we read buf as raw binary?
            buf = lzf.decompress(compressed_data, uncompressed_size)
            if len(buf) != uncompressed_size:
                raise Exception('Error decompressing data')
            # the data is stored field-by-field
            pcs_data = np.zeros(metadata['width'], dtype=dtype)
            ix = 0
            for dti in range(len(dtype)):
                dt = dtype[dti]
                bytess = dt.itemsize * metadata['width']
                column = np.fromstring(buf[ix:(ix + bytess)], dt)
                pcs_data[dtype.names[dti]] = column
                ix += bytess
            return pcs_data

        with open(fname, 'rb') as f:
            header = []
            while True:
                ln = f.readline().strip()
                header.append(ln)
                ln_str = ln.decode('ascii')
                if ln_str.startswith("DATA"):
                    metadata = parse_header(header)
                    dtype = _build_dtype(metadata)
                    break
            if metadata['data'] == 'ascii':
                pc_data = np.loadtxt(f, dtype=dtype, delimiter=' ')
                if include_ring:
                    x, y, z = pc_data['x'], pc_data['y'], pc_data['z']
                    i = pc_data['intensity']
                    r = pc_data['ring'].astype(np.float32)
                    pc = np.vstack((x, y, z, i, r)).transpose()
                    pc.dtype = np.float32
                    pc = pc.reshape(-1, 5)
                else:
                    x, y, z = pc_data['x'], pc_data['y'], pc_data['z']
                    i = pc_data['intensity']
                    pc = np.vstack((x, y, z, i)).transpose()
                    pc.dtype = np.float32
                    pc = pc.reshape(-1, 4)
                return pc
            elif metadata['data'] == 'binary':
                pc_data = parse_binary_pc_data(f, dtype, metadata)
            elif metadata['data'] == 'binary_compressed':
                pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
            else:
                print('GG: PCD DATA field is not "ascii", "binary" , "binary_compressed", try to add method')
                return 'CODE: 0x123'
            if from_bag:
                pc_raw = point_cloud(metadata, pc_data).pc_data
                x = pc_raw[:]["x"]
                y = pc_raw[:]["y"]
                z = pc_raw[:]["z"]
                if include_ring:
                    i = pc_raw[:]["intensity"]
                    r = pc_raw[:]["ring"]
                    pc = np.vstack((x, y, z, i, r)).transpose()
                    pc.dtype = np.float32
                    pc = pc.reshape(-1, 5)
                else:
                    i = np.zeros_like(x)
                    pc = np.vstack((x, y, z, i)).transpose()
                    pc.dtype = np.float32
                    pc = pc.reshape(-1, 4)
                return pc
            else:
                pc = point_cloud(metadata, pc_data).pc_data
                pc.dtype = np.float32
                if include_ring:
                    pc = pc.reshape(-1, 5)
                else:
                    pc = pc.reshape(-1, 4)
                return pc

    @staticmethod
    def save_points_as_bin(pc_np, filename):
        assert (len(pc_np.shape) == 2 and pc_np.shape[
            -1] == 4), "Function save_points_as_pcd: input data's shape is not compatible"

        pc_np.tofile(filename)


def show_pcd(dataPath, box=None):
    fileindex = sorted(os.listdir(dataPath))
    for File in fileindex:
        pc = point_cloud.load_pcd(os.path.join(dataPath, File))
        # if box is None:
        #     pcd_vispy(pc.pc_data)
        # else:
        #     pcd_vispy(pc.pc_data,boxes=box)


if __name__ == '__main__':
    pass
