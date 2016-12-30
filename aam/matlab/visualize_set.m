function visualize_set(I, pts, s, options)

batch_size = 24;
num_cols = 6;

N = length(s);
for k=1:ceil(N/batch_size)
    
    h_k = figure;
    for i=1:batch_size        
        idx = (k-1)*batch_size+i;
        if idx > N
            continue;
        end
        j = s(idx);
        subplot(batch_size/num_cols, num_cols, i);
        imshow(I{j}); hold on; plot(pts{j}(:,1), pts{j}(:,2), 'g.');
    end
    
    if options.saveit
        set(gcf, 'Position', get(0,'Screensize'));        
        savefig(h_k, [options.filename, num2str(k)]);
        saveas(h_k, [options.filename, num2str(k)], 'png');
    end
    
end

end