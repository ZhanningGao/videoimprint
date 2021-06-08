function mapn = circconv2(map,mask,R,E)

map = conv2(map,mask,'full');
mapn = map(R+1:end-R,R+1:end-R);
mapn(end-R+1:end,end-R+1:end) = mapn(end-R+1:end,end-R+1:end)+map(1:R,1:R);
mapn(end-R+1:end,1:R) = mapn(end-R+1:end,1:R)+map(1:R,R+E(2)+1:end);
mapn(1:R,end-R+1:end) = mapn(1:R,end-R+1:end)+map(R+E(1)+1:end,1:R);
mapn(1:R,1:R) = mapn(1:R,1:R)+map(R+E(1)+1:end,R+E(2)+1:end);
mapn(1:R,:) = mapn(1:R,:)+map(R+E(1)+1:end,R+1:R+E(2));
mapn(:,1:R) = mapn(:,1:R)+map(R+1:R+E(1),R+1+E(2):end);
mapn(end-R+1:end,:) = mapn(end-R+1:end,:)+map(1:R,R+1:R+E(2));
mapn(:,end-R+1:end) = mapn(:,end-R+1:end)+map(R+1:R+E(1),1:R);
end