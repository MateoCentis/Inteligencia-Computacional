function [F] = fdm2d_neumann(F,xnode,neighb,NEU)
    %NEUMANN
    M = s i z e (NEU, 1 ) ;
    for n = 1 : M
    P = NEU( n , 1 ) ;
     % centro
    S = neighb ( P , 1 ) ;
     % [ 1 ] sur
    E = neighb ( P , 2 ) ;
     % [ 2 ] este
    N = neighb ( P , 3 ) ;
     % [ 3 ] norte
    W = neighb ( P , 4 ) ;
     % [ 4 ] oeste
    q = NEU( n , 2 ) ;
     % c a l o r impuesto
    i f (E == −1)
    dx = abs ( xnode (W, 1 ) − xnode ( P , 1 ) ) ;
    end
    i f (NEU( n , 3 ) == 2 )
     % [ 2 ]F (P) = F (P) − 2∗q / dx ;
    end
    end
    end