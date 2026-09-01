function [wsout, wvout, m,stats] = tkpm(t, Mfun, u, tol, m, iom, options)
% tkpm
%
% Evaluates a linear combination of T-functions acting on input vectors
% via an adaptive Krylov subspace method.
% Let the input matrix be
%
%       u = [u1, u2, ..., u_p],
%
% where each column is one vector. Then tkpm computes two quantities,
% ws and wv, given by
%
%   ws = T_0(t^2 M) u1 + t T_1(t^2 M) u2
%       + sum_{j=1}^{p-2} t^(j+1) T_{j+1}(t^2 M) u_{j+2},
%
%   wv = T_0(t^2 M) u2 - t M T_1(t^2 M) u1
%       + sum_{j=1}^{p-2} t^(j) T_{j}(t^2 M) u_{j+2}.
%
% In the compact recurrence form, the same outputs are written as
%
%   ws(t_k+tau_k) = sum_{i=1}^{p-2} tau_k^(i-1)/(i-1)! omega_i
%       + tau_k^(p-2) T_{p-2}(tau_k^2 M) omega_{p-1}
%       + tau_k^(p-1) T_{p-1}(tau_k^2 M) omega_p,
%
%   wv(t_k+tau_k) = sum_{i=2}^{p-2} tau_k^(i-2)/(i-2)! omega_i
%       + tau_k^(p-3) T_{p-3}(tau_k^2 M) omega_{p-1}
%       + tau_k^(p-2) T_{p-2}(tau_k^2 M) omega_p.
%
% Here omega_1 = ws(t_k), omega_2 = wv(t_k), and
%
%   omega_i = -M omega_{i-2}
%             + sum_{ell=0}^{p-i} t_k^ell/ell! u_{i+ell},
%             i = 3,...,p.
%
% The vector omega_{p-1} is handled by the odd Krylov basis, while
% omega_p is handled by the even Krylov basis.
%
%   [wsout, wvout, m, stats] = tkpm(t, Mfun, u, tol, m, iom, options)
%
% Inputs:
%   t      - Array of time values (nonzero, strictly increasing)
%   Mfun   - Function handle (or matrix) for the action of M
%   u      - Matrix whose columns are the input vectors. (The expected
%            number of T functions is p-1.)
%   tol    - Tolerance for convergence.
%   m      - Initial estimate of the Krylov subspace size.
%   iom    - Incomplete orthogonalization length.
%   options - (Optional) structure with fields:
%             .max_m          : Maximum Krylov subspace dimension [default: 100]
%             .safety_factors : Structure with fields gamma and delta 
%                              [default: gamma = 0.8, delta = 1.4]
%
% Outputs:
%   wsout  - Matrix of solutions computed at times t.
%   wvout  - Secondary output computed from a modified combination.
%   m      - Final Krylov subspace size used.
%   stats  - [# substeps, # rejected steps, # Krylov steps, # phi function call]
%
% This function forces (p-1) to be even by padding u if necessary.

%
% Modified by [Van Hoang Nguyen], Spring 2025


% ========================================================================
% Part 1. Set default options
% ========================================================================
% If the optional structure "options" is not provided, create an empty one.
% Then set default values for the maximum Krylov dimension and the safety
% factors used in the adaptive strategy.

if nargin < 7
     options = struct();
end

if ~isfield(options, 'max_m')           
    options.max_m = 100; 
end

if ~isfield(options, 'safety_factors')
    options.safety_factors = struct('gamma', 0.8, 'delta', 1.2);
end

% ========================================================================
% Part 2. Get dimensions and check the input time array
% ========================================================================
% n is the dimension of each vector.
% p is the number of input vectors stored in u.
% N is the number of output times.

[n, p] = size(u);

N = length(t);

% The code assumes that the first output time is nonzero.
if any(t == 0)
    error('tkpm: all entries of t must be nonzero');
end

if any(diff(t) <= 0)
    error('tkpm: entries of t must be strictly increasing');
end


% ========================================================================
% Part 3. Preallocate output arrays
% ========================================================================
% wsout and wvout store the two computed outputs at all requested times.
% The kth column corresponds to the output at t(k).

wsout = zeros(n, N);
wvout = zeros(n, N);


% ========================================================================
% Part 4. Force p-1 to be even by padding u
% ========================================================================
% The second-order formulation separates the recurrence into odd and even
% parts. To make the indexing consistent, the code forces p-1 to be even.
% If needed, zero columns are appended to u. These zero columns do not
% change the value of the requested matrix-function combination.

if p == 1
    error('tkpm: error, u has at least 2 vectors');
elseif p == 2
    u = [u, zeros(n,1), zeros(n,1), zeros(n,1)];
    p = p + 3;
elseif p == 3
    u = [u, zeros(n,1), zeros(n,1)];
    p = p + 2;
elseif mod(p-1, 2) ~= 0
    % If p-1 is odd, add one zero column so that p-1 becomes even.
    u = [u, zeros(n,1)];
    p = p + 1;
end

% ========================================================================
% Part 5. Persistent arrays for Krylov bases and recurrence vectors
% ========================================================================
% V_even stores the Krylov basis for the even contribution.
% V_odd  stores the Krylov basis for the odd contribution.
% int    stores the internal recurrence vectors.
%
% These arrays are persistent, so MATLAB keeps them in memory between calls.
% This avoids repeated memory allocation when the function is called many
% times with the same dimensions.

persistent V_even V_odd int nnze

mmax_val = options.max_m;
mnew = m;

% Allocate or reallocate the even Krylov basis if needed.
if isempty(V_even) || numel(V_even) ~= n*(mmax_val+1)
    V_even = zeros(n, mmax_val+1);
end

% Allocate or reallocate the odd Krylov basis if needed.
if isempty(V_odd) || numel(V_odd) ~= n*(mmax_val+1)
    V_odd = zeros(n, mmax_val+1);
end

% Allocate or reallocate the internal recurrence array if needed.
if isempty(int) || numel(int) ~= n*p
    int = zeros(n, p);
end


% ========================================================================
% Part 6. Convert Mfun to a function handle and estimate matrix cost
% ========================================================================
% If Mfun is given as a matrix, convert it to a function handle so that the
% code can always use Mfun(v) to compute M*v.
%
% nnze is used later as a rough cost estimate in the adaptive strategy.

if isnumeric(Mfun)
    if isempty(nnze)
        nnze = nnz(Mfun);
    end
    Mfun = @(x) Mfun*x;
else
    nnze = 7*n;
end


% ========================================================================
% Part 7. Initialize counters, solution variables, and adaptivity state
% ========================================================================
% step    counts accepted substeps.
% reject  counts rejected attempts.
% krystep counts Krylov matrix-vector products.
% phis    counts calls to the projected phi-function routine.

step = 0; 
reject = 0; 
krystep = 0; 
Ts = 0;

% Extract the safety factors for adaptivity.
gamma = options.safety_factors.gamma;
delta = options.safety_factors.delta;


% Initialize the two evolving quantities.
% In the second-order setting, these are the two outputs propagated together.
ws = u(:,1);
wv = u(:,2);

% Current internal time.
tnow = 0;

% Krylov dimension counter.
j = 0;

% Initial time step.
tau = abs(t(1));


% ------------------------------------------------------------------------
% Initialize the adaptive state.
% ------------------------------------------------------------------------
% These variables store information from the previous accepted/rejected
% attempts and are passed to update_adaptivity.

oldtau = NaN;
oldm = NaN;
omega = NaN;
orderold = true;
kestold = true;

% ireject counts rejected attempts for the current substep.
ireject = 0; 

% ========================================================================
% Part 8. Main time loop
% ========================================================================
% The outer loop goes through the requested output times.
% The inner loop advances the solution from the current internal time tnow
% to the current output time tout, possibly using several adaptive substeps.

for itout = 1:N
    % Current output time.
    tout = abs(t(itout));
    while tnow < tout
        % Do not allow the current substep to go beyond tout.
        tau = min(tout-tnow, tau);
        % Reset happy-breakdown flags for the current attempted substep.
        happy = false;
        happy_odd = false;
        happy_even = false;
        % ----------------------------------------------------------------
        % Part 8.1. Initialize a new Krylov substep
        % ----------------------------------------------------------------
        % If j == 0, then the code starts a new substep.
        % It builds the internal recurrence vectors and determines whether
        % the odd part, the even part, or both parts require Krylov
        % approximation.

        if j == 0

            % Projected Hessenberg matrices for the even and odd Krylov bases.
            % Extra rows/columns are included for the augmented phi-function
            % computation.
            H_even = zeros(mmax_val+(p+3)/2, mmax_val+(p+3)/2); 
            H_odd = zeros(mmax_val+(p+3)/2, mmax_val+(p+3)/2);

            % Build time-dependent Taylor-like coefficients involving tnow.
            x = [zeros(1, p-1), cumprod([1, tnow./(1:p-1)])];

            % Build an index matrix used to combine columns of u with the
            % shifted polynomial coefficients.
            cidx = (0:p-1)'; 
            ridx = p:-1:1;
            idx = cidx(:, ones(p,1)) + ridx(ones(p,1),:);

            % Form the shifted input-vector combination at the current time.
            up = u * x(idx);
            
            % ------------------------------------------------------------
            % Compute internal recurrence vectors.
            % ------------------------------------------------------------
            % The first two recurrence vectors are the current outputs.
            % Then the recurrence
            %
            %   int(:,i+2) = -M*int(:,i) + up(:,i+2)
            %
            % generates the remaining vectors. This recurrence is the source
            % of the odd/even splitting.

            int(:,1) = ws;
            int(:,2) = wv;

            for i = 1:p-2
                int(:, i+2) = -Mfun(int(:, i)) + up(:, i+2);
            end
            

            % The final two recurrence vectors define the starting 
            % directions for the two Krylov corrections.
            %
            %   omega_{p-1} = int(:,end-1)   -> odd Krylov correction,
            %   omega_p     = int(:,end)     -> even Krylov correction.
            %
            % Their norms are used to normalize the first Krylov basis vectors:

            beta_odd  = norm(int(:,end-1));  % norm of omega_{p-1}
            beta_even = norm(int(:,end));    % norm of omega_p

            % ------------------------------------------------------------
            % Initialize the proper Krylov basis or handle the zero case.
            % ------------------------------------------------------------
            % If both beta values are zero, no Krylov correction is needed.
            % If only one is nonzero, initialize only that Krylov basis.
            % If both are nonzero, initialize both bases.

            if beta_odd == 0 && beta_even == 0

                reject = reject + ireject;
                step = step + 1;
                tau = tout - tnow;

                ws = ws + int(:,2:p-2) * cumprod(tau./(1:(p-3))');
                wv = wv + int(:,3:p-2) * cumprod(tau./(1:(p-4))');
                break;

            elseif beta_odd ~= 0 && beta_even == 0

                V_odd(:,1) = int(:, end-1) ./ beta_odd;

            elseif beta_odd == 0 && beta_even ~= 0

                V_even(:,1) = int(:, end) ./ beta_even;

            else

                V_odd(:,1) = int(:, end-1) ./ beta_odd;
                V_even(:,1) = int(:, end) ./ beta_even;

            end
        end


        % ----------------------------------------------------------------
        % Part 8.2. Branch selection based on beta_odd and beta_even
        % ----------------------------------------------------------------
        % The algorithm chooses one of three Krylov paths:
        %
        %   1. only the odd Krylov correction is needed,
        %   2. only the even Krylov correction is needed,
        %   3. both odd and even Krylov corrections are needed.
        %
        % If both beta values are zero, the polynomial recurrence alone is
        % used and no Krylov basis is built.

        if beta_odd == 0 && beta_even == 0

            % No Krylov correction is needed.
            reject = reject + ireject;
            step = step + 1;
            tau = tout - tnow;

            ws = ws + int(:,2:p-2) * cumprod(tau./(1:(p-3))');
            wv = wv + int(:,3:p-2) * cumprod(tau./(1:(p-4))');

            % Update time and reset counters.
            tnow = tnow + tau;
            j = 0;
            ireject = 0;

            break;        

        elseif beta_odd ~= 0 && beta_even == 0

            % ----------------------------------------------------------------
            % Part 8.3. Branch 1: only beta_odd is nonzero
            % ----------------------------------------------------------------
            % Build the odd Krylov basis using incomplete orthogonalization.

            while j < m

                 % Increase Krylov dimension and compute M*V_odd(:,j).
                 j = j+1;

                 if (norm(V_odd(:,j)) > 0)
                    vv_odd = Mfun(V_odd(:,j));
                 else
                    vv_odd = V_odd(:,j);
                 end      


                 % Incomplete orthogonalization:
                 % Orthogonalize only against the most recent iom vectors.
                 for i = max(1,j-iom):j
                    H_odd(i,j) = V_odd(:,i)'*vv_odd;
                    vv_odd = vv_odd - H_odd(i,j)*V_odd(:,i);
                 end

                 krystep = krystep + 1;
                 s_odd = norm(vv_odd);


                % Happy breakdown: no new meaningful Krylov vector is needed.
                if s_odd < tol
                    happy = true;
                    tau = tout - tnow;
                    break;
                end

                % Store Hessenberg entry and normalized next basis vector.
                H_odd(j+1, j) = s_odd;
                V_odd(:, j+1) = vv_odd / s_odd;

            end


            % Backup H_odd in case this attempted step is rejected.
            H_odd_b = H_odd;

            % Augment H_odd for the projected T-function computation.
            H_odd(1, j+1) = 1;

            for i = 1:(p-1)/2
                H_odd(j+i, j+i+1) = 1;
            end

            % Store the residual coefficient h, then remove it from H_odd.
            h = H_odd(j+1, j);
            H_odd(j+1, j) = 0;

            % Compute projected phi-function matrices.
            [F0_odd, F1_odd, hnorm_odd] = ...
                phi0phi1m_norm(tau^2 * H_odd(1:j+(p-1)/2, 1:j+(p-1)/2));

            Ts = Ts + 1;

            % Estimate the local Krylov error from the projected quantities.
            err = max(abs(beta_odd*h*F0_odd(j, j+(p-1)/2)), ...
                      abs(tau*beta_odd*h*F1_odd(j, j+(p-1)/2)));

            hnorm = hnorm_odd;

            oldomega = omega;
            omega = tout*err/(tau*tol);
                  
            % Estimate temporal order of accuracy
            if ( (m == oldm) && (tau ~= oldtau) && (ireject >= 1) )
               order = max(1, log(omega/oldomega)/log(tau/oldtau));
               orderold = false;
            elseif ( orderold || (ireject == 0) )
               orderold = true;
               % order = j/2;
               order = j/4;
            else
               orderold = true;
            end
          
            % Estimate convergence factor, k, for varying Krylov subspace size
            if ( (m ~= oldm) && (tau == oldtau) && (ireject >= 1) )
               kest = max(1.1, (omega/oldomega)^(1/(oldm-m)));
               kestold = false;
            elseif ( kestold || (ireject == 0) )
               kestold = true;
               kest = 2;
            else
               kestold = true;
            end
    
            % This if block is the main difference between fixed and variable m
            oldtau = tau; 
            oldm = m;
            if happy   
               % Happy breakdown; wrap up
               omega = 0;
               taunew = tau;
               mnew = m;
    
            elseif ( (j == options.max_m) && (omega > delta) )
             
               % Krylov subspace too small and stepsize too large
               taunew = tau*(omega/gamma)^(-1/order);
             
            else
    
               % Determine optimal tau and m
               tauopt = tau*(omega/gamma)^(-1/order);
               mopt = max(1, ceil(j+log(omega/gamma)/log(kest)));
               nom = 8 + (max(log(sqrt(hnorm)/49.8589), 0) / log(4));
               cost = @(M_val, T_val) (2*nnze - 1)*(p - 1) + (2*n - 1)*(p - 1)*(p - 2)/2 ...
                       + (2*p-1)*n + 2*((2*M_val*nnze+(11*M_val+1)*n)+...
                       nom*((M_val+(p+3)/2)^3)/3)* ceil((tout-tnow)/T_val);
    
               % Determine whether to vary tau or m
               if (cost(j, tauopt) < cost(mopt, tau))
                  taunew = tauopt;
                  mnew = m;
               else
                  mnew = mopt;
                  taunew = tau;
               end             
            end
            
            if (omega <= delta)           
                % Step accepted.
                reject = reject + ireject;
                step = step + 1;
    
                % Polynomial part of the update.
                ups = ws + int(:,2:p-2)*cumprod(tau*1./(1:p-3)');
                upv = wv + int(:,3:p-2)*cumprod(tau*1./(1:p-4)');
        
                % Accepted update when only the odd correction is present.
                F0_odd(j+1,j+(p-1)/2-1) =  h*F0_odd(j,j+(p-1)/2);
                F1_odd(j+1,j+(p-1)/2-1) =  h*F1_odd(j,j+(p-1)/2);
    
                ws = ups + tau*(-1)^((p-1)/2-1)*beta_odd* ...
                     V_odd(:,1:j+1)*F1_odd(1:j+1,j+(p-1)/2-1);                
    
                wv = upv + (-1)^((p-1)/2-1)*beta_odd* ...
                     V_odd(:,1:j+1)*F0_odd(1:j+1,j+(p-1)/2-1); 
                % Advance time and reset local counters.
                tnow = tnow + tau;
                j = 0;
                ireject = 0;
            else
                H_odd = H_odd_b;
                ireject = ireject + 1;
             
            end  
    
            tau = max(tau/5, min(2*tau, taunew));
    
            m = max(1, min(mmax_val, ...
                max(floor(3/4*m), min(mnew, ceil(4/3*m)))));

        elseif beta_odd == 0 && beta_even ~=0

            % ----------------------------------------------------------------
            % Part 8.4. Branch 2: only beta_even is nonzero
            % ----------------------------------------------------------------
            % Build the even Krylov basis using incomplete orthogonalization.

            while j < m

                 % Increase Krylov dimension and compute M*V_even(:,j).
                 j = j+1;

                 if (norm(V_even(:,j)) > 0)
                    vv_even = Mfun(V_even(:,j));
                 else
                    vv_even = V_even(:,j);
                 end      


                 % Incomplete orthogonalization:
                 % Orthogonalize only against the most recent iom vectors.
                 for i = max(1,j-iom):j
                    H_even(i,j) = V_even(:,i)'*vv_even;
                    vv_even = vv_even - H_even(i,j)*V_even(:,i);
                 end

                 krystep = krystep + 1;
                 s_even = norm(vv_even);


                % Happy breakdown.
                if s_even < tol
                    happy = true;
                    tau = tout - tnow;
                    break;
                end

                % Store Hessenberg entry and next normalized basis vector.
                H_even(j+1, j) = s_even;
                V_even(:, j+1) = vv_even / s_even;

            end


            % Backup H_even in case this attempted step is rejected.
            H_even_b = H_even;

            % Augment H_even for the projected phi-function computation.
            H_even(1, j+1) = 1;

            for i = 1:(p+1)/2
                H_even(j+i, j+i+1) = 1;
            end

            % Store residual coefficient h, then remove it from H_even.
            h = H_even(j+1, j);
            H_even(j+1, j) = 0;

            % Compute projected phi-function matrices.
            [F0_even, F1_even, hnorm_even] = ...
                phi0phi1m_norm(tau^2 * H_even(1:j+(p+1)/2, 1:j+(p+1)/2));

            Ts = Ts + 1;

            % Estimate local Krylov error for the even contribution.
            err = max(abs(beta_even*h*F0_even(j, j+(p+1)/2)), ...
                      abs(tau*beta_even*h*F1_even(j, j+(p-1)/2)));

            hnorm = hnorm_even;
          % Compute error per unit step
            oldomega = omega;
            omega = tout*err/(tau*tol);
                      
            % Estimate temporal order of accuracy
            if ( (m == oldm) && (tau ~= oldtau) && (ireject >= 1) )
               order = max(1, log(omega/oldomega)/log(tau/oldtau));
               orderold = false;
            elseif ( orderold || (ireject == 0) )
               orderold = true;
               % order = j/2;
               order = j/4;
            else
               orderold = true;
            end
              
            % Estimate convergence factor, k, for varying Krylov subspace size
            if ( (m ~= oldm) && (tau == oldtau) && (ireject >= 1) )
               kest = max(1.1, (omega/oldomega)^(1/(oldm-m)));
               kestold = false;
            elseif ( kestold || (ireject == 0) )
               kestold = true;
               kest = 2;
            else
               kestold = true;
            end
            
            % This if block is the main difference between fixed and variable m
            oldtau = tau; 
            oldm = m;
            if happy    
              % Happy breakdown; wrap up
               omega = 0;
               taunew = tau;
               mnew = m;
            elseif ( (j == options.max_m) && (omega > delta) )
                 
               % Krylov subspace too small and stepsize too large
               taunew = tau*(omega/gamma)^(-1/order);
                 
            else
            
               % Determine optimal tau and m
               tauopt = tau*(omega/gamma)^(-1/order);
               mopt = max(1, ceil(j+log(omega/gamma)/log(kest)));
               nom = 8 + (max(log(sqrt(hnorm)/49.8589), 0) / log(4));
               cost = @(M_val, T_val) (2*nnze - 1)*(p - 1) + (2*n - 1)*(p - 1)*(p - 2)/2 ...
                       + (2*p-1)*n + 2*((2*M_val*nnze+(11*M_val+1)*n)+...
                       nom*((M_val+(p+3)/2)^3)/3)* ceil((tout-tnow)/T_val);
            
               % Determine whether to vary tau or m
               if (cost(j, tauopt) < cost(mopt, tau))
                  taunew = tauopt;
                  mnew = m;
               else
                  mnew = mopt;
                  taunew = tau;
               end                 
            end
            if (omega <= delta)       

                % Step accepted.
                reject = reject + ireject;
                step = step + 1;
    
                % Polynomial part of the update.
                ups = ws + int(:,2:p-2)*cumprod(tau*1./(1:p-3)');
                upv = wv + int(:,3:p-2)*cumprod(tau*1./(1:p-4)');
    
                F0_even(j+1,j+(p+1)/2-1) =  h*F0_even(j,j+(p+1)/2);
                F1_even(j+1,j+(p-1)/2-1) =  h*F1_even(j,j+(p-1)/2);
    
                ws = ups + ((-1)^((p+1)/2-1))*beta_even* ...
                     V_even(:,1:j+1)*F0_even(1:j+1,j+(p+1)/2-1);                
    
                wv = upv + tau*((-1)^((p-1)/2-1))*beta_even* ...
                     V_even(:,1:j+1)*F1_even(1:j+1,j+(p-1)/2-1);  
    
                % Advance time and reset local counters.
                tnow = tnow + tau;
                j = 0;
                ireject = 0;

            else
                H_even = H_even_b;
                ireject = ireject + 1;
             
            end  

        tau = max(tau/5, min(2*tau, taunew));

        m = max(1, min(mmax_val, ...
            max(floor(3/4*m), min(mnew, ceil(4/3*m)))));
       
        else
            % ----------------------------------------------------------------
            % Part 8.5. Branch 3: both beta_odd and beta_even are nonzero
            % ----------------------------------------------------------------
            % Build both odd and even Krylov bases, then combine the two
            % projected error estimates and updates.

            j_odd = 0;

            % Build odd Krylov basis.
            while j_odd < m

                 j_odd = j_odd+1;

                 if (norm(V_odd(:,j_odd)) > 0)
                    vv_odd = Mfun(V_odd(:,j_odd));
                 else
                    vv_odd = V_odd(:,j_odd);
                 end      


                 % Incomplete orthogonalization for the odd basis.
                 for i = max(1,j_odd-iom):j_odd
                    H_odd(i,j_odd) = V_odd(:,i)'*vv_odd;
                    vv_odd = vv_odd - H_odd(i,j_odd)*V_odd(:,i);
                 end

                 krystep = krystep + 1;
                 s_odd = norm(vv_odd);


                % Happy breakdown for the odd Krylov basis.
                if s_odd < tol
                    happy_odd = true;
                    tau = tout - tnow;
                    break;
                end

                H_odd(j_odd+1, j_odd) = s_odd;
                V_odd(:, j_odd+1) = vv_odd / s_odd;

            end

            j_even = 0;

            % Build even Krylov basis.
            while j_even < m

                 j_even = j_even+1;

                 if (norm(V_even(:,j_even)) > 0)
                    vv_even = Mfun(V_even(:,j_even));
                 else
                    vv_even = V_even(:,j_even);
                 end      

                 % Incomplete orthogonalization for the even basis.
                 for i = max(1,j_even-iom):j_even
                    H_even(i,j_even) = V_even(:,i)'*vv_even;
                    vv_even = vv_even - H_even(i,j_even)*V_even(:,i);
                 end

                 krystep = krystep + 1;
                 s_even = norm(vv_even);


                % Happy breakdown for the even Krylov basis.
                if s_even < tol
                    happy_even = true;
                    tau = tout - tnow;
                    break;
                end

                H_even(j_even+1, j_even) = s_even;
                V_even(:, j_even+1) = vv_even / s_even;

            end

            if happy_odd && happy_even
                happy = true;
            end

            % Use the larger of the two Krylov dimensions for adaptivity.
            j = max(j_even, j_odd);

            % Backup both Hessenberg matrices in case of rejection.
            H_odd_b = H_odd;
            H_even_b = H_even;

            % Augment both projected Hessenberg matrices.
            H_odd(1, j_odd+1) = 1;
            H_even(1, j_even+1) = 1;

            for i = 1:(p+1)/2
                H_odd(j_odd+i, j_odd+i+1) = 1;
                H_even(j_even+i, j_even+i+1) = 1;
            end

            % Store residual coefficients.
            h_odd = H_odd(j_odd+1, j_odd);
            h_even = H_even(j_even+1, j_even);

            % Remove residual entries before projected phi computation.
            H_odd(j_odd+1, j_odd) = 0;
            H_even(j_even+1, j_even) = 0;
            
            % Compute projected phi-function matrices for odd and even parts.
            [F0_odd, F1_odd, hnorm_odd] = ...
                phi0phi1m_norm(tau^2 * H_odd(1:j_odd+(p-1)/2, 1:j_odd+(p-1)/2));
            
            [F0_even, F1_even, hnorm_even] = ...
                phi0phi1m_norm(tau^2 * H_even(1:j_even+(p+1)/2, 1:j_even+(p+1)/2));
            
            Ts = Ts + 2;

            % % Combined local error estimate from odd and even contributions.
            err = max([abs(beta_odd*h_odd*F0_odd(j_odd, j_odd+(p-1)/2)), ...
                       abs(tau*beta_odd*h_odd*F1_odd(j_odd, j_odd+(p-1)/2)), ...
                       abs(beta_even*h_even*F0_even(j_even, j_even+(p+1)/2)), ...
                       abs(tau*beta_even*h_even*F1_even(j_even, j_even+(p-1)/2))]);

            hnorm = max(hnorm_odd, hnorm_even);
            oldomega = omega;
            omega = tout*err/(tau*tol);
                  
            % Estimate temporal order of accuracy
            if ( (m == oldm) && (tau ~= oldtau) && (ireject >= 1) )
               order = max(1, log(omega/oldomega)/log(tau/oldtau));
               orderold = false;
            elseif ( orderold || (ireject == 0) )
               orderold = true;
               % order = j/2;
               order = j/4;
            else
               orderold = true;
            end
          
            % Estimate convergence factor, k, for varying Krylov subspace size
            if ( (m ~= oldm) && (tau == oldtau) && (ireject >= 1) )
               kest = max(1.1, (omega/oldomega)^(1/(oldm-m)));
               kestold = false;
            elseif ( kestold || (ireject == 0) )
               kestold = true;
               kest = 2;
            else
               kestold = true;
            end
    
            % This if block is the main difference between fixed and variable m
            oldtau = tau; 
            oldm = m;
            if happy
    
               % Happy breakdown; wrap up
               omega = 0;
               taunew = tau;
               mnew = m;
    
            elseif ( (j == options.max_m) && (omega > delta) )
             
               % Krylov subspace too small and stepsize too large
               taunew = tau*(omega/gamma)^(-1/order);
             
            else
    
               % Determine optimal tau and m
               tauopt = tau*(omega/gamma)^(-1/order);
               mopt = max(1, ceil(j+log(omega/gamma)/log(kest)));
               nom = 64/3 + 4*(max(log(sqrt(hnorm)/49.8589), 0) / log(4));
               cost = @(M_val, T_val) (2*nnze - 1)*(p - 1) + (2*n - 1)*(p - 1)*(p - 2)/2 ...
                       + (2*p-1)*n + 2*((2*M_val*nnze+(11*M_val+1)*n)+...
                       nom*((M_val+(p+3)/2)^3)/3)* ceil((tout-tnow)/T_val);
    
               % Determine whether to vary tau or m
               if (cost(j, tauopt) < cost(mopt, tau))
                  taunew = tauopt;
                  mnew = m;
               else
                  mnew = mopt;
                  taunew = tau;
               end
             
            end
     
        if (omega <= delta)       

            % Step accepted.
            reject = reject + ireject;
            step = step + 1;

            % Polynomial part of the update.
            ups = ws + int(:,2:p-2)*cumprod(tau*1./(1:p-3)');
            upv = wv + int(:,3:p-2)*cumprod(tau*1./(1:p-4)');
    
            % Accepted update when both odd and even corrections appear.
            F0_odd(j_odd+1,j_odd+(p-1)/2-1) = ...
                h_odd*F0_odd(j_odd,j_odd+(p-1)/2);

            F1_odd(j_odd+1,j_odd+(p-1)/2-1) = ...
                h_odd*F1_odd(j_odd,j_odd+(p-1)/2);

            F0_even(j_even+1,j_even+(p+1)/2-1) = ...
                h_even*F0_even(j_even,j_even+(p+1)/2);

            F1_even(j_even+1,j_even+(p-1)/2-1) = ...
                h_even*F1_even(j_even,j_even+(p-1)/2);


            % Slice bases once.
            Vodd  = V_odd(:,  1:j_odd+1);
            Veven = V_even(:, 1:j_even+1);

            % Slice projected phi-function vectors once.
            idx1  = j_odd  + (p-3)/2 ;
            idx2 = j_even + (p-1)/2 ;

            F1o = F1_odd(1:j_odd+1,   idx1 );
            F0o = F0_odd(1:j_odd+1,   idx1 );
            F1e = F1_even(1:j_even+1, idx2-1);
            F0e = F0_even(1:j_even+1, idx2);


            % Build a block basis so the final correction uses two
            % large matrix-vector products instead of four.
            Vblock = [Vodd, Veven];  

            % Coefficient vectors for ws and wv.
            F_ws = [ tau*((-1)^((p-3)/2))*beta_odd  * F1o ;
                     ((-1)^((p-1)/2))*beta_even * F0e ];

            F_wv = [ ((-1)^((p-3)/2))*beta_odd  * F0o ;
                     tau*((-1)^((p-3)/2))*beta_even * F1e ];


            % Final accepted updates.
            ws = ups + Vblock * F_ws;
            wv = upv + Vblock * F_wv;

            % Advance time and reset local counters.
            tnow = tnow + tau;
            j = 0;
            ireject = 0;
        else
            H_odd = H_odd_b;
            H_even = H_even_b;
            ireject = ireject + 1;
         
        end  
        tau = max(tau/5, min(2*tau, taunew));

        m = max(1, min(mmax_val, ...
            max(floor(3/4*m), min(mnew, ceil(4/3*m)))));


        end
    end

    stats = {step, reject, krystep, Ts};

    % Store the computed outputs at the current requested output time.
    wsout(:,itout) = ws;
    wvout(:,itout) = wv;

end
end 