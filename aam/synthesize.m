function [s, g] = synthesize(model, bs, bg, k)
s = model.shape.P * bs + model.shape.x;
g = model.texture{k}.P * bg + model.texture{k}.x;
end