% Copyright 2019 Yılmaz Kaya, Cüneyt ÖZDEMİR /  SIIRT UNIVERSITY / TURKEY

function [motifs]=motifyeni(data,Pboyut)

S=data;

PB=Pboyut; % Pencere boyu
P=perms(1:PB); % pencere boyuna bağlı tüm permütasyonlar
Pc=zeros(size(P,1),1);

for i=1:length(S)-PB+1
    a=S(i:i+PB-1); % pencere boyutu kadar sinyalden alınan parça 
    b=indisBul(a); % alınan parçanın örüntüsünün elde edilmesi

    cd=sum(abs(P-repmat(b(:,2)',size(P,1),1)),2); % hangi patterne uyduğu bulunuyor
    patindx = find(cd==0);
    Pc(patindx)=Pc(patindx)+1; % bulunan patter sayacı 1 arttırılıyor

end

motifs=Pc';
end
