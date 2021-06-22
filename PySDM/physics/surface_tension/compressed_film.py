from PySDM.physics import constants as const#, si

class CompressedFilm:
    @staticmethod
    def sigma(T, v_wet, v_dry):
        # TODO #223 - these parameters should be passed from Settings
        # chemical parameters
        delta_min = 0.2e-9  # minimum organic film thickness [nm]
        sgm_org = const.sgm_w*0.5      # organic surface tension [N m-1]
        f_org = .5

        # convert volumes to diameters
        d_coat = 2 * ((3*v_dry) / (4*const.pi))**(1/3)
        d_seed = (d_coat**3 - f_org * d_coat**3)**(1/3)
        d_wet = 2 * ((3*v_wet) / (4*const.pi))**(1/3)
        
        # calculate the total volume of organic
        v_beta = v_dry - const.pi/6 * d_seed**3
        
        # calculate the min shell volume, v_delta by Eq. (S3.4.3)
        v_delta = const.pi/6 * (d_wet**3 - (d_wet**3 - 2*delta_min)**3)
        
        # calculate the coverage parameter using Eq. (S3.4.2)
        c_beta = min(v_beta/v_delta, 1)
        
        # calculate sigma
        sgm = (1-c_beta)*const.sgm_w + c_beta*sgm_org
        return sgm
