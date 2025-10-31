d = 100;
B = 20;
B1 = 20;
B2 = 5;
KK = 1;
N1all = linspace(1.5,3,B1);
N2all = linspace(2,4,B);
R = lispace(0.5,1,5);
I = 25000;
s = zeros(B1,B,B2,I);
Sigma = eye(d);
Mu = zeros(1,d);
%%
for k = 1:B2
    for j = 1:B
        for t = 1:B1
            Call = zeros(I,d,d);
            Muall = zeros(d,I);
            N1 = floor(10^N1all(t));
            N2 = floor(10^N2all(j));
            r = R(k);
            aa = mvnrnd(Mu,real(Sigma),floor(N1));
            mp = mean(aa);
            sp = mpower(cov(aa),1);
            s(t,j,k,1) = real(wd(Mu,mp,Sigma,sp));
            %s(hh,1) = real(wd(Mu,mp,Sigma,sp));
            Call(1,:,:) = sp;
            for i = 2:I
                strcat('N1 = ',num2str(t), ' N2 = ', num2str(j), ' r=', num2str(k), ' i = ', num2str(i) )
                A = zeros(N2 + N1,d);
                L = min(KK,i-1);
                h = floor(N2/L);
                NN = N2 - (L-1)*h;
                for kk = 1:L
                    if kk == L
                        A(1+(kk-1)*h:N2,:) = mvnrnd(Muall(:,i-L-1+ kk),real(squeeze(Call(i-L-1+ kk,:,:))),NN);
                    else
                        A(1+ (kk-1)*h:kk*h,:) = mvnrnd(Muall(:,i-L-1+ kk),real(squeeze(Call(i-L-1+ kk,:,:))),h);
                    end
                 end
                 A(N2+1:end, :) = mvnrnd(Mu,real(Sigma),N1);
                 mp = mean(A);
                 sp = r*mpower(cov(A),1);
                 s(t,j,k,i) =  real(wd(Mu,mp,Sigma,sp/r));
                 Call(i,:,:) = sp;
                 Muall(:,i) = mp;
            end
        end
    end
    
end            
%%
fidinv = zeros(1,2500);
for i = 1:2500
    i
    N1 = i + 10;
    for r = 1:1
       aa = mvnrnd(Mu,real(Sigma),N1);
       mp = mean(aa);
       sp = mpower(cov(aa),1);
       fidinv(i) = fidinv(i) + real(wd(Mu,mp,Sigma,sp));
    end
end
fidinv = fidinv/1;
%%
ss = squeeze(mean(s(:,:,:,1000:end),4));
Ns = zeros(B1,B,B2);
for i = 1:B1
    i
    for j = 1:B
        for k = 1:B2
            [~,f] = min((abs( ss(i,j,k) - fidinv )));
            Ns(i,j,k) = f+10;
        end
    end
end
%%
for i = 1:B1
        Ns(i,:,:) = Ns(i,:,:)/(floor(10^N1all(i)));
end
%%
pcolor(N1all,N2all,log(fliplr(flipud(ss(:,:,1))')));
colorbar
%%
contourf(N1all(1:end),N2all,(((Ns'))), 30);
xlabel('${log(N_r)}$', 'interpreter','latex','FontSize', 15)
ylabel('${log(N_s)}$', 'interpreter','latex','FontSize', 15)
title('${r = 1}$', 'interpreter','latex','FontSize', 15)
colormap jet
colorbar
%caxis([0 3])
%%
plot(Ns)
