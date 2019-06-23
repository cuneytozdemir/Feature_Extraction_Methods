% Copyright 2019 Yılmaz Kaya, Cüneyt ÖZDEMİR /  SIIRT UNIVERSITY / TURKEY

function [binlist,binmap]=vectmap(data,komsuluk,alfa,beta)

if ~exist('komsuluk','var'), komsuluk = 1; end
if ~exist('alfa','var'), alfa = 1; end
if ~exist('beta','var'), beta = 1; end

binlist=[];
binmap=[];
l=length(data);

komsuluk=komsuluk*beta;

for i=1:l
    kon(1:2*komsuluk/beta)=data(i);
    dat=zeros(1,2*komsuluk/beta);
    if (i<=komsuluk+alfa-1)
        if (i>alfa)
            dat(komsuluk/beta:-1:ceil((komsuluk-i+alfa+1)/beta))=data((i-alfa):-beta:1);
        end
        dat((komsuluk/beta+1):2*komsuluk/beta)=data((i+alfa):beta:(i+komsuluk+alfa-1));
    elseif ((i+komsuluk+alfa-1)>l)
        dat(komsuluk/beta:-1:1)=data((i-alfa):-beta:(i-komsuluk-alfa+1));
        if (i+alfa<=l)
            dat((komsuluk/beta+1):ceil((komsuluk+1+l-i-alfa)/beta))=data((i+alfa):beta:l);
        end
    elseif ((i>komsuluk+alfa-1) && i<=(l-komsuluk-alfa+1))
        dat(komsuluk/beta:-1:1)=data((i-alfa):-beta:(i-komsuluk-alfa+1)); 
        dat((komsuluk/beta+1):2*komsuluk/beta)=data((i+alfa):beta:(i+komsuluk+alfa-1));
    end
    binlist(i,:)=(kon<dat);
end
binmap=bi2de(binlist);
end
