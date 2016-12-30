function news = scale_shape(s, sz)

pts = reshape(s, 2, [])';
N = size(pts, 1);

minx = min(pts(:,1)); maxx = max(pts(:,1));
miny = min(pts(:,2)); maxy = max(pts(:,2));

center = 0.5 * [minx+maxx, miny+maxy];
xrange = maxx-minx;
yrange = maxy-miny;

factor = sz / max(xrange, yrange);

pts = (pts - repmat(center, N, 1)) * factor + 0.5 * repmat([250, 250], N, 1);

news = reshape(pts', [], 1);

end