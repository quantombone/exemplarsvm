function [negatives,vals,pos] = find_set_membership(m)
%Given the objects inside m.svids, parse their curids to see which
%of the following three sets they belong to

if ~isfield(m,'bg_string1')
  bg = get_pascal_bg('trainval');
else
  bg = get_pascal_bg(m.bg_string1, m.bg_string2);
end

bgtrain = get_pascal_bg('train',['-' m.cls]);
bgval = get_pascal_bg('val',['-' m.cls]);
fg = get_pascal_bg('trainval',m.cls);

bgids = zeros(length(bg),1);
aa = find(ismember(bg,bgtrain));
bgids(aa) = 1;
aa = find(ismember(bg,bgval));
bgids(aa) = 2;
aa = find(ismember(bg,fg));
bgids(aa) = 3;

ids = cellfun(@(x)x.curid,m.model.svids);
%Train only with negatives
negatives = find(ismember(ids,(find(bgids==1))));
vals = find(ismember(ids,(find(bgids==2))));
pos = find(ismember(ids,(find(bgids==3))));

fprintf(1,'Sets NEG,VAL,POS=%d,%d,%d\n',...
        length(negatives),length(vals),length(pos));

