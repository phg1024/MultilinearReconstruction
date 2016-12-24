function [r, g, b] = split_channels(I)
r = I(:,:,1);
g = I(:,:,2);
b = I(:,:,3);
end