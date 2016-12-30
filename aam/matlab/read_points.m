function points = read_points(filename)
fid = fopen(filename, 'r');

npoints = fscanf(fid, '%d', 1);
points = fscanf(fid, '%f');

points = reshape(points, 2, []);
points = points';

fclose(fid);

end