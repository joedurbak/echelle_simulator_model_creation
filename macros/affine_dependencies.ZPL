nfield = NFLD()
nconfig = NCON()
nwaves = NWAV()
nsurf = NSUR()
!echellesurface = 38
maxfield = MAXF()

PRINT "config,wave,wavelength,fieldx,fieldy,x,y"
!PRINT "config,wave,wavelength,field,fieldx,fieldy,x,y"
!,z,l,m,n"

FOR config, 1, nconfig, 1
    
    SETCONFIG config
    
    FOR wave, 1, nwaves, 1
    
        PWAV wave
        wavelength = WAVL(wave)
        
        FOR fieldx, -1, 1, 1
        
            FOR fieldy, -1, 1, 1

                RAYTRACE fieldx, fieldy, 0, 0, PWAV()
                x = RAYX(nsurf)
                y = RAYY(nsurf)
                !z = RAYZ(nsurf)
                !l = RAYL(nsurf)
                !m = RAYM(nsurf)
                !n = RAYN(nsurf)
                PRINT config, ",", wave, ",", wavelength, ",", fieldx, ",", fieldy, ",", x, ",", y  
                !PRINT config, ",", wave, ",", wavelength, ",", field, ",", fieldx, ",", fieldy, ",", x, ",", y  
                ! , ",", z, ",", l, ",", m, ",", n

            NEXT fieldy

        NEXT fieldx

    NEXT wave
    
NEXT config
