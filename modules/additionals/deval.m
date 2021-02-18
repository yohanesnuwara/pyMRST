function [ val ] = deval( s,t )
  %% This function was written by Andres Codas in MRST forum
  %% Visit the link to forum: https://groups.google.com/g/sintef-mrst/c/te22P6tLLB4/m/vhCMHcqyAgAJ
val = interp1(s.x,s.y,t(:)');
end
