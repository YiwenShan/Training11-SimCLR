clear;
N = 256; % batch size
s_ik = 2.*(rand(2*N-2,1) - 0.5); % [-1, 1]
s_il = -1:0.05:1;

figure;
for tau = [0.1 0.2 0.3 0.5 1]
    f = 1./(tau.*(1 + exp(-s_il/tau) .* sum( exp(s_ik./tau) )));
    plot(s_il, f, '-','linewidth',2); hold on;
end
legend('$\tau=0.1$','$\tau=0.2$','$\tau=0.3$','$\tau=0.5$','$\tau=1$','interpreter','latex',...
    'location','northwest','fontsize',16);
xlabel('$\mathbf{z}_{i}^{\top} \mathbf{z}_{l}$','interpreter','latex','fontsize',16);
ylabel('$\frac{\partial \mathcal{L}(\mathbf{z}_{i},\mathbf{z}_{j})}{\partial \mathbf{z}_{l}}$','interpreter','latex','fontsize',16);
ylim([0.0, 0.55]);hold off;
