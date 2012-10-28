function subinds = get_subinds(covstruct_full,hg_size)


hg_full = zeros(covstruct_full.hg_size(1), covstruct_full.hg_size(2), covstruct_full.hg_size(3));

offset = [0 0];
offset = round((covstruct_full.hg_size(1:2) - covstruct_full.hg_size(1:2))/2);

hg_full(offset(1) + (1:hg_size(1)),...
        offset(2) + (1:hg_size(2)),:) = 1;

subinds = find(hg_full);
