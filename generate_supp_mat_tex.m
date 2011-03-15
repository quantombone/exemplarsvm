%% Generate supplemental materials tex (see also supplemental.tex)

classes = setdiff(VOCopts.classes,'person');

for q = 1:length(classes)
  sprintf('stuff_%s.tex',classes{q})
  fid = fopen(sprintf('stuff_%s.tex',classes{q}),'w');
  fprintf(fid,'\\begin{figure*}\n');
  fprintf(fid,'\\begin{centering}\n');
  

  fprintf(fid,'\\begin{tabular}{cc}\n');
  cls = classes{q};
  c = 1;
  for i = 1:3
    fprintf(fid,['\\includegraphics[width=.42\\linewidth]{../%s_%05d}' ...
                 ' \\hspace{.2in} & \\hspace{.2in} \\includegraphics[width=.42\\linewidth]{../%s_%05d} \\\\\n'],cls,c,cls,c+1);
    c = c + 2;
  end
  fprintf(fid,'\\end{tabular}\n');
  fprintf(fid,'\\caption{Top $6$ Detection results for category %s.}\n',cls);
  fprintf(fid,'\\label{fig:stuff-%s}\n',cls);
  fprintf(fid,'\\end{centering}\n');
  fprintf(fid,'\\end{figure*}\n');

  fclose(fid);
end
