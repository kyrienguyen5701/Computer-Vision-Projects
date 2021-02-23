const MODELS_PATH = '/static/models';
const FACE_DESCRIPTORS_KEY = 'face_descriptors';
const FACE_LABEL = 'ME';

const REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
]

const DEFAULT_CROP_SIZE = [96, 112]

function descriptors2Json(descriptors) {
    json = {
        label: descriptors.label,
        descriptors: descriptors.descriptors.map((d) =>　Array.from(d))
    }
    return JSON.stringify(json);
}

function json2Descriptor(json) {
    const descriptors = json.descriptors.map((d) => {
        return new Float32Array(d);
    });
    return new faceapi.LabeledFaceDescriptors(json.label, descriptors);
}

Array.prototype.add = function (arr) {
    var sum = [];
    if (arr != null && this.length == arr.length) {
        for (var i = 0; i < arr.length; i++) {
            sum.push(this[i] + arr[i]);
        }
    }

    return sum;
}

function norm(a) {
    var sum = 0.0
    for(var i=0; i<a.length; i++) {
        sum += a[i]*a[i]
    }
    sum = Math.sqrt(sum)
    return array_div_scalar(a, sum)
}

function array_div_scalar(a,b) {
    var c = []
    for(var i=0; i < a.length; i++) {
        c.push(a[i]/b)
    }
    return c
}

function get_five_facial_points(landmarks) {
    const leftEyeX = (landmarks[37].x + landmarks[40].x)/2
    const leftEyeY = (landmarks[37].y + landmarks[40].y)/2
    const leftEye = [leftEyeX, leftEyeY]
    const rightEyeX = (landmarks[43].x + landmarks[46].x)/2
    const rightEyeY = (landmarks[43].y + landmarks[46].y)/2
    const rightEye = [rightEyeX, rightEyeY]
    return [leftEye, rightEye, [landmarks[30].x, landmarks[30].y], 
            [landmarks[48].x, landmarks[48].y], [landmarks[54].x, landmarks[54].y]]
}

function get_reference_facial_points(output_size=null,
                                    inner_padding_factor=0.0,
                                    outer_padding=[0, 0],
                                    default_square=false) {
    var tmp_5pts = REFERENCE_FACIAL_POINTS
    var tmp_crop_size = DEFAULT_CROP_SIZE
    if (default_square) {
        var max_crop_size = Math.max(tmp_crop_size[0], tmp_crop_size[1])
        var size_diff_x = max_crop_size - tmp_crop_size[0]
        var size_diff_y = max_crop_size - tmp_crop_size[1]
        tmp_5pts[0][0] += size_diff_x/2
        tmp_5pts[0][1] += size_diff_y/2
        tmp_5pts[1][0] += size_diff_x/2
        tmp_5pts[1][1] += size_diff_y/2
        tmp_5pts[2][0] += size_diff_x/2
        tmp_5pts[2][1] += size_diff_y/2
        tmp_5pts[3][0] += size_diff_x/2
        tmp_5pts[3][1] += size_diff_y/2
        tmp_crop_size[0] += size_diff_x
        tmp_crop_size[1] += size_diff_y
    }

    if (output_size &&
        output_size[0] == tmp_crop_size[0] &&
        output_size[1] == tmp_crop_size[1]) {
        return tmp_5pts
    }
    if (inner_padding_factor == 0 && outer_padding[0] == 0 && outer_padding[0] == 1) {
        if (!output_size) {
            return tmp_5pts
        } else {
            return null
        }
    }
    if (inner_padding_factor < 0 || inner_padding_factor > 1.0) {
        return null
    }
    if ((inner_padding_factor > 0 || outer_padding[0] > 0 || outer_padding[1] > 0) && (!output_size)) {
        output_size[0] = tmp_crop_size[0] * parseInt(1 + inner_padding_factor * 2) + outer_padding[0]
        output_size[1] = tmp_crop_size[1] * parseInt(1 + inner_padding_factor * 2) + outer_padding[1]
        console.log('deduced from paddings, output_size = ', output_size)
    }
    if (outer_padding[0] >= output_size[0] || outer_padding[1] >= output_size[1]) {
        return null
    }
    if (inner_padding_factor > 0) {
        size_diff_x = tmp_crop_size[0] * inner_padding_factor * 2
        size_diff_y = tmp_crop_size[1] * inner_padding_factor * 2
        tmp_5pts[0][0] += size_diff_x/2
        tmp_5pts[0][1] += size_diff_y/2
        tmp_5pts[1][0] += size_diff_x/2
        tmp_5pts[1][1] += size_diff_y/2
        tmp_5pts[2][0] += size_diff_x/2
        tmp_5pts[2][1] += size_diff_y/2
        tmp_5pts[3][0] += size_diff_x/2
        tmp_5pts[3][1] += size_diff_y/2
        tmp_crop_size[0] += Math.round(size_diff_x)
        tmp_crop_size[1] += Math.round(size_diff_y)
    }
    var size_bf_outer_pad = [0, 0]
    size_bf_outer_pad[0] = output_size[0] - outer_padding[0] * 2
    size_bf_outer_pad[1] = output_size[1] - outer_padding[1] * 2
    if (size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]) {
        return null
    }
    const scale_factor = parseFloat(size_bf_outer_pad[0]) / tmp_crop_size[0]
    tmp_5pts[0][0] += scale_factor
    tmp_5pts[0][1] *= scale_factor
    tmp_5pts[1][0] *= scale_factor
    tmp_5pts[1][1] *= scale_factor
    tmp_5pts[2][0] *= scale_factor
    tmp_5pts[2][1] *= scale_factor
    tmp_5pts[3][0] *= scale_factor
    tmp_5pts[3][1] *= scale_factor
    tmp_crop_size[0] = size_bf_outer_pad[0]
    tmp_crop_size[1] = size_bf_outer_pad[1]
    var reference_5point = tmp_5pts
    reference_5point[0][0] += outer_padding[0]
    reference_5point[0][1] += outer_padding[1]
    reference_5point[1][0] += outer_padding[0]
    reference_5point[1][1] += outer_padding[1]
    reference_5point[2][0] += outer_padding[0]
    reference_5point[2][1] += outer_padding[1]
    reference_5point[3][0] += outer_padding[0]
    reference_5point[3][1] += outer_padding[1]
    tmp_crop_size[0] = output_size[0]
    tmp_crop_size[1] = output_size[1]

    return reference_5point
}

function alignFace(image, landmarks) {
    var default_square = true
    var inner_padding_factor = 0.25
    var outer_padding = [0, 0]
    var output_size = [112, 112]
    var crop_size = [112, 112]
    var reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)
    var dst = warp_and_crop_face(image, landmarks, reference_5pts, crop_size)
    return dst
}

function warp_and_crop_face(src,
                            facial_pts,
                            reference_pts=null,
                            crop_size=[96, 112]) {
    if (!reference_pts) {
        if (crop_size[0] == 96 && crop_size[1] == 112) {
            reference_pts = REFERENCE_FACIAL_POINTS
        } else {
            default_square = false
            inner_padding_factor = 0
            outer_padding = [0, 0]
            output_size = crop_size

            reference_pts = get_reference_facial_points(output_size,
                                                        inner_padding_factor,
                                                        outer_padding,
                                                        default_square)
        }
    }
    const tfm = similarity_transform(facial_pts, reference_pts)
    var dst = new cv.Mat()
    var dsize = new cv.Size(112, 112)
    const M = cv.matFromArray(2, 3, cv.CV_32FC1, tfm)
    cv.warpAffine(src, dst, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar())
    M.delete();
    return dst
}

function transpose(a) {
    const rows = a.length
    const cols = a[0].length
    var b = []
    for(var j=0; j < cols; j++) {
        var row = []
        for (var i=0; i <rows; i++) {
            row.push(a[i][j])
        }
        b.push(row)
    }
    return b
}

function mat_mul(a, b) { // a: (2,5) , b: (5,2)
    var mat = []
    const rows_a = a.length
    const rows_b = b.length
    const cols_b = b[0].length
    for(var i=0; i < rows_a; i++) {
        var row = []
        for(var j=0; j < cols_b; j++) {
            var e = 0.0
            for(var k=0; k < rows_b; k++) {
                e += a[i][k] * b[k][j]
            }
            row.push(e)
        }
        mat.push(row)
    }
    // c[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0] + a[0][3]*b[3][0] + a[0][4]*b[4][0]
    // c[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1] + a[0][3]*b[3][1] + a[0][4]*b[4][1]
    // c[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0] + a[1][3]*b[3][0] + a[1][4]*b[4][0]
    // c[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1] + a[1][3]*b[3][1] + a[1][4]*b[4][1]
    return mat
}

function div_scalar(a, b) {
    // console.log('a: ', a)
    var c = [[0.0, 0.0], [0.0, 0.0]]
    c[0][0] = a[0][0] / b
    c[0][1] = a[0][1] / b
    c[1][0] = a[1][0] / b
    c[1][1] = a[1][1] / b
    return c
}

function swap(mat, row1, row2, col) { 
    for (var i = 0; i < col; i++) { 
        var temp = mat[row1][i]; 
        mat[row1][i] = mat[row2][i]; 
        mat[row2][i] = temp; 
    }
    return mat
} 

function matrix_rank(mat) {
    var rank = 2
    for (var row = 0; row < rank; row++) {
        if (mat[row][row]) { 
            for (var col = 0; col < 2; col++) { 
                if (col != row) { 
                    var mult = mat[col][row] / mat[row][row]; 
                    for (var i = 0; i < rank; i++) {
                        mat[col][i] -= mult * mat[row][i]; 
                    }
                } 
            } 
        } 
  
        else { 
            var reduce = true; 
  
            for (var i = row + 1; i < 2;  i++) { 
                if (mat[i][row]) { 
                    mat = swap(mat, row, i, rank); 
                    reduce = false; 
                    break; 
                } 
            } 
  
            if (reduce) { 
                rank--; 
                for (var i = 0; i < 2; i ++) {
                    mat[i][row] = mat[i][rank]; 
                }
            }   
            row--; 
        }
    } 
    return rank; 
}

function det(mat) {
    return mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]
}

function diagonal_array(d) {
    var mat = []
    for (var i = 0; i < d.length; i ++) {
        var row = []
        for (var j = 0; j < d.length; j++) {
            if (j == i) {
                row.push(d[i])
            } else {
                row.push(0.0)
            }
        }
        mat.push(row)
    }
    return mat
}

function sum(mat) {
    const rows = mat.length
    const cols = mat[0].length
    var sum = 0.0
    for (var i=0; i < rows; i++) {
        for (var j=0; j < cols; j++) {
            sum += mat[i][j]
        }
    }
    return sum
}

function sum_row(r) {
    var sum = 0.0
    const rows = r.length
    for (var i=0; i < rows; i++) {
        sum += r[i]
    }
    return sum
}

function mean(mat, axis=0) {
    const rows = mat.length
    const cols = mat[0].length
    var means = []
    if (axis == 0) {
        for (var j=0; j<cols; j++) {
            var sum = 0.0
            for (var i=0; i <rows; i++) {
                sum += mat[i][j]
            }
            means.push(sum/rows)
        }
    } else {
        for (var i=0; i<rows; i++) {
            var sum = 0.0
            for (var j=0; j<cols; j++) {
                sum += mat[i][j]
            }
            means.push(sum/cols)
        }
    }
    return means
}

function variance(mat, axis=0) {
    var variances = []
    const rows = mat.length
    const cols = mat[0].length
    const means = mean(mat, axis)
    if (axis == 0) {
        for (var j=0; j<cols; j++) {
            var v = 0.0
            for (var i=0; i <rows; i++) {
                const diff = mat[i][j] - means[j]
                v += diff*diff
            }
            variances.push(v/rows)
        }
    } else {
        for (var i=0; i<rows; i++) {
            var v = 0.0
            for (var j=0; j<cols; j++) {
                const diff = mat[i][j] - means[i]
                v += diff*diff
            }
            variances.push(v/cols)
        }
    }
    return variances
}

function copy_mat(mat) {
    var ret = []
    const rows = mat.length
    const cols = mat[0].length
    for (var i=0; i<rows; i++) {
        var row = []
        for (var j=0; j<cols; j++) {
            row.push(mat[i][j])
        }
        ret.push(row)
    }
    return ret
}

function dot(a, b) {
    var sum = 0.0
    for (var i=0; i<a.length; i++) {
        sum += a[i]*b[i]
    }
    return sum
}

function similarity_transform(src_pts, dst_pts) {
    var src_mean = [0.0, 0.0]
    src_mean[0] = (src_pts[0][0] + src_pts[1][0] + src_pts[2][0] + src_pts[3][0] + src_pts[4][0]) / 5
    src_mean[1] = (src_pts[0][1] + src_pts[1][1] + src_pts[2][1] + src_pts[3][1] + src_pts[4][1]) / 5

    var dst_mean = [0.0, 0.0]
    dst_mean[0] = (dst_pts[0][0] + dst_pts[1][0] + dst_pts[2][0] + dst_pts[3][0] + dst_pts[4][0]) / 5
    dst_mean[1] = (dst_pts[0][1] + dst_pts[1][1] + dst_pts[2][1] + dst_pts[3][1] + dst_pts[4][1]) / 5

    // console.log('src_pts: ', src_pts)
    // console.log('dst_pts: ', dst_pts, 'dst_mean: ', dst_mean)

    var src_demean = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    src_demean[0][0] = src_pts[0][0] - src_mean[0]
    src_demean[1][0] = src_pts[1][0] - src_mean[0]
    src_demean[2][0] = src_pts[2][0] - src_mean[0]
    src_demean[3][0] = src_pts[3][0] - src_mean[0]
    src_demean[4][0] = src_pts[4][0] - src_mean[0]
    src_demean[0][1] = src_pts[0][1] - src_mean[1]
    src_demean[1][1] = src_pts[1][1] - src_mean[1]
    src_demean[2][1] = src_pts[2][1] - src_mean[1]
    src_demean[3][1] = src_pts[3][1] - src_mean[1]
    src_demean[4][1] = src_pts[4][1] - src_mean[1]

    var dst_demean = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    dst_demean[0][0] = dst_pts[0][0] - dst_mean[0]
    dst_demean[1][0] = dst_pts[1][0] - dst_mean[0]
    dst_demean[2][0] = dst_pts[2][0] - dst_mean[0]
    dst_demean[3][0] = dst_pts[3][0] - dst_mean[0]
    dst_demean[4][0] = dst_pts[4][0] - dst_mean[0]
    dst_demean[0][1] = dst_pts[0][1] - dst_mean[1]
    dst_demean[1][1] = dst_pts[1][1] - dst_mean[1]
    dst_demean[2][1] = dst_pts[2][1] - dst_mean[1]
    dst_demean[3][1] = dst_pts[3][1] - dst_mean[1]
    dst_demean[4][1] = dst_pts[4][1] - dst_mean[1]

    // console.log('src_demean: ', src_demean)
    // console.log('dst_demean: ', dst_demean)

    var A = mat_mul(transpose(dst_demean), src_demean)
    A = div_scalar(A, 5)
    // console.log('A: ', A)
    det_A = det(A)
    var d = [1.0, 1.0]
    if (det_A < 0) {
        d[1] = -1.0
    }
    T = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    const { u, v, q } = SVDJS.SVD(A)
    // console.log('u: ', u)
    // console.log('v: ', v)
    // console.log('q: ', q)
    var u1 = [[0.0, 0.0], [0.0, 0.0]]
    u1[0][0] = u[0][1]
    u1[0][1] = u[0][0]
    u1[1][0] = u[1][1]
    u1[1][1] = u[1][0]
    var v1 = [[0.0, 0.0], [0.0, 0.0]]
    v1[0][0] = v[0][1]
    v1[0][1] = v[0][0]
    v1[1][0] = v[1][1]
    v1[1][1] = v[1][0]
    var q1 = [0.0, 0.0]
    q1[0] = q[1]
    q1[1] = q[0]
    // console.log('u: ', u1)
    // console.log('v: ', v1)
    // console.log('q: ', q1)
    const rank = matrix_rank(copy_mat(A))
    // console.log('rank: ', rank)
    if(rank == 0) {
        return null
    } else if (rank == 1) {
        if (det(u1)*det(v1) > 0) {
            const uv = mat_mul(u1,v1)
            T[0][0] = uv[0][0]
            T[0][1] = uv[0][1]
            T[1][0] = uv[1][0]
            T[1][1] = uv[1][1]
        } else {
            s = d[1]
            d[1] = -1.0
            const diagonal = diagonal_array(d)
            const udv = mat_mul(mat_mul(u1,diagonal), v1)
            T[0][0] = udv[0][0]
            T[0][1] = udv[0][1]
            T[1][0] = udv[1][0]
            T[1][1] = udv[1][1]
            d[1] = s
        }
    } else {
        const diagonal = diagonal_array(d)
        const udv = mat_mul(mat_mul(u1,diagonal), v1)
        T[0][0] = udv[0][0]
        T[0][1] = udv[0][1]
        T[1][0] = udv[1][0]
        T[1][1] = udv[1][1]
    }
    const src_variance = sum_row(variance(src_demean))
    // console.log('source variance: ', src_variance)
    const sd = q1[0]*d[0] + q1[1]*d[1]
    const scale =  1.0 / src_variance * sd
    // console.log('scale: ',scale)
    var T2 = [[0.0, 0.0], [0.0, 0.0]]
    T2[0][0] = T[0][0]
    T2[0][1] = T[0][1]
    T2[1][0] = T[1][0]
    T2[1][1] = T[1][1]
    T2_src_mean = [0.0, 0.0]
    T2_src_mean[0] = scale*(T2[0][0]*src_mean[0] + T2[0][1]*src_mean[1])
    T2_src_mean[1] = scale*(T2[1][0]*src_mean[0] + T2[1][1]*src_mean[1])
    // console.log('dst mean: ', dst_mean)
    // console.log('T2_src_mean: ', T2_src_mean)
    T[0][2] = dst_mean[0] - T2_src_mean[0]
    T[1][2] = dst_mean[1] - T2_src_mean[1]
    T[0][0] *= scale
    T[0][1] *= scale
    T[1][0] *= scale
    T[1][1] *= scale
    // console.log('T: ', T)

    var ret = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ret[0] = T[0][0]
    ret[1] = T[0][1]
    ret[2] = T[0][2]
    ret[3] = T[1][0]
    ret[4] = T[1][1]
    ret[5] = T[1][2]
    return ret
}

function getAxes(yaw, pitch, roll, tdx, tdy, size=100) {
    const pi = Math.PI;
    const sin = Math.sin;
    const cos = Math.cos;

    yaw = -yaw * pi / 180
    pitch = yaw >= 0 ? -pitch * pi / 180 : pitch * pi / 180
    roll = roll * pi / 180

    // x-axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    // y-axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    // z-axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    return {
        x_axis: {
            x: x1,
            y: y1,
            c: 'rgb(255, 0, 0)'
        },
        y_axis: {
            x: x2,
            y: y2,
            c: 'rgb(0, 255, 0)'
        },
        z_axis: {
            x: x3,
            y: y3,
            c: 'rgb(0, 0, 255)'
        }
    }
}