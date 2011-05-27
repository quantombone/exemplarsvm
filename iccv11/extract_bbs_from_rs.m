function [bboxes] = extract_bbs_from_rs(rs, index)
%% Given a resultstruct 'rs' from a localizemeHOG function and an
%% image index, convert the results into one large result matrix
%% which stores the bb's

%% bboxes(i,:) = [b1 b2 b3 b4 ? exid flip scale o1 o2 index score]
%% bboxes(i,:) = [1  2  3  4  5 6    7    8     9  10 11    12];
%% Tomasz Malisiewicz (tomasz@cmu.edu)

bboxes = cell(length(rs.id_grid),1);
for j = 1:length(rs.id_grid)
  if length(rs.id_grid{j})==0
    continue
  end
  bbs = cellfun2(@(x)x.bb,rs.id_grid{j});
  bbs = cat(1,bbs{:});
  bbs(:,5:12) = 0;
  bbs(:,6) = j;
  bbs(:,7) = cellfun(@(x)x.flip,[rs.id_grid{j}])';
  bbs(:,8) = cellfun(@(x)x.scale,[rs.id_grid{j}])';
  bbs(:,9) = cellfun(@(x)x.offset(1),[rs.id_grid{j}])';
  bbs(:,10) = cellfun(@(x)x.offset(2),[rs.id_grid{j}])';
  bbs(:,11) = index;
  bbs(:,12) = rs.score_grid{j}';
  
  bboxes{j} = bbs;
end

bboxes = cat(1,bboxes{:});
