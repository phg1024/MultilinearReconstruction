%% align shape s to the reference shape s0
function t = align_shape(s, s0)

p0 = reshape(s0, 2, [])';
p = reshape(s, 2, [])';

tform = estimateGeometricTransform(p, p0, 'similarity');

t = transformPointsForward(tform, p);
t = reshape(t', [], 1);

end