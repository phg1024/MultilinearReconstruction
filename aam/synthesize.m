function [s, g] = synthesize(model, bs, bg)
s = model.shape.P * bs + model.shape.x;
g = model.texture.P * bg + model.texture.x;
end