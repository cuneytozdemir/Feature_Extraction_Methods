% Copyright 2019 Yılmaz Kaya, Cüneyt ÖZDEMİR /  SIIRT UNIVERSITY / TURKEY

function [acidata]=Angle(data)
data=data';
acilar=[];
N=size(data,1);
for i=1:N-2
    P=data(i:i+2,:);
    x=i:i+2;
    P=[x' P ];
   
    dd=det([P(2,:)-P(1,:);P(3,:)-P(2,:)]);
    dtt=dot(P(2,:)-P(1,:),P(3,:)-P(2,:));
    ang = atan2(abs(dd),dtt);
    ac=round(ang*180/pi)+180;
    acilar=[acilar ac];
end

acidata=acilar;
end



   

