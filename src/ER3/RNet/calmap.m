function [map, mapall] = calmap(prob, test_y)

map = [];
 for i = 1:size(test_y,2)
     iClass = prob(:,i);
     iY = test_y(:,i);
     [Y I]=sort(iClass,'descend');
     iY = iY(I)';
     count = 0;
     ap =0;
     for j=1:length(iY)
         if(iY(j)==1)
             count = count+1;
             ap = ap + count/j;  
         end  
     end
     map =[map; ap/count];
 end
 mapall = map;
 map = mean(map);
