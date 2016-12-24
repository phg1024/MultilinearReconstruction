function draw_shape(s, color)
if nargin < 2
    color = 'g.';
end

pts = reshape(s, 2, [])';
plot(pts(:,1), pts(:,2), color);
end