function val = Ker(x,z,tao)
% this function is to implement the kernel function

dis = (x - z) * (x - z)';
val = exp(-1 /(2* tao^2) * dis );