
#imports
import numpy as np



import importlib

import sys
sys.path.append('../')

import utils
from utils import *
importlib.reload(utils)

import contractive_factor
from contractive_factor import *
importlib.reload(contractive_factor)





###########################################     ISTA     #########################################################
####################################################################################################################
#we will wrte a function to compute the linear operation P from teh linear operators W and G
#input: input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, eta
#output: output_image
def P_operator(input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, \
               eta):
    output_image = input_image - eta * A_function_adjoint(A_function(input_image, **A_kwargs), **A_adjoint_kwargs)
    output_image = denoiser(output_image, **denoiser_kwargs)
    return output_image

# we will write a function as adjoint of P operator
def P_operator_adjoint(input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, \
                A_adjoint_kwargs, eta):
    output_image = denoiser(input_image, **denoiser_kwargs)
    output_image = output_image - eta * A_function_adjoint(A_function(output_image, **A_kwargs), **A_adjoint_kwargs)
    return output_image


#we will write a function D to compute the linear operator D
#the linear operation is just a multiplication of the input image by a matrix of the same size as image
#and point wise multiplication
#input: input_image, D_matrix
#output: output_image
#processing: D_matrix * input_image
def D(input_image, D_matrix, D_power):
    #we will raise each element of D_matrix to the power D_power
    D_matrix = np.power(D_matrix, D_power)
    #multiply the input image by D_matrix
    output_image = D_matrix*input_image
    return output_image



#we will define a function to compute D_half_P_operator_D_half_inverse
#input: input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, eta, D_matrix, D_power                                                          
def D_half_P_D_half_inverse(input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, eta, D_matrix):
    #call D
    output_image = D(input_image=input_image, D_matrix=D_matrix, D_power=-1/2)
    #call P
    output_image = P_operator(input_image=output_image, denoiser=denoiser, denoiser_kwargs=denoiser_kwargs, \
        A_function=A_function, A_kwargs=A_kwargs, A_function_adjoint=A_function_adjoint, A_adjoint_kwargs=\
            A_adjoint_kwargs, eta=eta)
    #call D
    output_image = D(input_image=output_image, D_matrix=D_matrix, D_power=1/2)
    return output_image

#we will define a function to compute D_half_P_operator_D_half_inverse_adjoint
#input: input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, eta, D_matrix, D_power
def D_half_P_D_half_inverse_adjoint(input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, eta, D_matrix):
    #call D
    output_image = D(input_image=input_image, D_matrix=D_matrix, D_power=1/2)
    #call P
    output_image = P_operator_adjoint(input_image=output_image, denoiser=denoiser, denoiser_kwargs=denoiser_kwargs, \
        A_function=A_function, A_kwargs=A_kwargs, A_function_adjoint=A_function_adjoint, A_adjoint_kwargs=\
            A_adjoint_kwargs, eta=eta)
    #call D
    output_image = D(input_image=output_image, D_matrix=D_matrix, D_power=-1/2)
    return output_image


#we will write a function to get the gradient of the forward model at x
#gradient is : forward_operator_adjoint ( forward_operator(x) - b)
#input: A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, x, b
#output: gradient
def get_gradient(A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, x, b):
    #apply A_function to x with A_kwargs
    Ax = A_function(x, **A_kwargs)
    #subtract b from Ax
    Ax_minus_b = Ax - b
    #apply A_function_adjoint to Ax_minus_b with A_adjoint_kwargs
    gradient = A_function_adjoint(Ax_minus_b, **A_adjoint_kwargs)
    #return gradient
    return gradient


def contraction_factor_P(image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, \
                A_adjoint_kwargs, eta, max_iterations=1000, \
                            norm_tolerance=1e-8, dot_tolerance=1e-12, plot= True, verbose = True):
    P_kwargs = {'denoiser': denoiser, 'denoiser_kwargs': denoiser_kwargs, 'A_function': A_function, \
                    'A_kwargs': A_kwargs, 'A_function_adjoint': A_function_adjoint, 'A_adjoint_kwargs': \
                        A_adjoint_kwargs, 'eta': eta}
    contraction_factor, _ , _ = power_method_for_images_non_symmetric ( functional = P_operator, \
        functional_adjoint = P_operator_adjoint, image_height = image.shape[0], image_width = image.shape[1], \
        args_dict_functional = P_kwargs, args_dict_functional_adjoint = P_kwargs, \
            max_iterations = max_iterations, norm_tolerance = norm_tolerance, dot_tolerance = dot_tolerance, \
                plot = plot, verbose = verbose )
    
    return contraction_factor



#we will implement the ISTA algorithm
#input: image, denoiser, denoiser_kwargs, iterations_to_fix_W,  iterations, step_size, tol, verbose
#output: x_k_plus_1
def ISTA(image, x_k, b,  denoiser, denoiser_kwargs, iterations_to_fix_W, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs,\
          i, eta, power_method_max_iterations, calculate_contractive_factor = False):
    #calculate the gradient
        #we will call the function get_gradient, based on the forward model
        
        #we will call the get_gradient function, we have to build the arguments
        gradf = get_gradient(A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, x_k, b)
        # gradf = A_kwargs[str(A_function.__name__)+'_matri x']*(x_k - b)
        #calculate the update
        grad_descent = x_k - eta * gradf
        
        

        #add one keyword parameter to the denoiser "guide_image" , value grad_descent
        #this we will use to fix the guide image in the denoiser, i.e. parameter to the denoiser
        #we have if iterations_to_fix_W = -1, we will use the original image as the guide image
        # if iterations_to_fix_W == 0:
        #     denoiser_kwargs['guide_image'] = image
        # #if iterations_to_fix_W is nay value > 0, we will adapt the guide image
        # elif iterations_to_fix_W > 0:
        #     #current iteartion < iterations_to_fix_W, we will use the gradient
        #     if i < iterations_to_fix_W:
        #         denoiser_kwargs['guide_image'] = grad_descent
        # #anything else will be error
        # else:
        #     raise ValueError('iterations_to_fix_W should be a non-negative integer')
        if denoiser.__name__ == 'DSG_NLM' or denoiser.__name__ == 'NLM':
            if iterations_to_fix_W == 0:
                denoiser_kwargs['guide_image'] = image
            #if iterations_to_fix_W is nay value > 0, we will adapt the guide image
            elif iterations_to_fix_W > 0:
                #current iteartion < iterations_to_fix_W, we will use the gradient
                if i < iterations_to_fix_W:
                    denoiser_kwargs['guide_image'] = grad_descent
            #anything else will be error
            else:
                raise ValueError('iterations_to_fix_W should be a non-negative integer')
            #denoiser ----------------
            #check that the keword argumnt in dictionary 'guide_image' is present and not None
            assert 'guide_image' in denoiser_kwargs.keys() and denoiser_kwargs['guide_image'] is not None

        #else if denoiser is denoise_tv_chambolle
        #estimate sigma, and add weight parameter
        elif denoiser.__name__ == 'denoise_tv_chambolle':
            pass
            # #estimate sigma
            # sigma = estimate_sigma(x_k - z_k, multichannel=False)
            # #add weight parameter
            # denoiser_kwargs['weight'] = 0.1*sigma

            # denoiser_kwargs['weight'] = 0.1

            # #debug ----------------
            # #print weight
            # print('weight: ', denoiser_kwargs['weight'])
            # #print max and min of x_k-z_k
            # print('max of x_k-z_k: ', np.max(x_k-z_k))
            # print('min of x_k-z_k: ', np.min(x_k-z_k))

            # #max and min of x_k
            # print('max of x_k: ', np.max(x_k))
            # print('min of x_k: ', np.min(x_k))

            # #max and min of z_k
            # print('max of z_k: ', np.max(z_k))
            # print('min of z_k: ', np.min(z_k))
            # #debug ----------------

        #assert that anything else is error, denoiser not implamented
        else:
            raise ValueError('Denoiser not implemented')
            
        ## Contractive factor -------------------------------------------------------------------------
        if calculate_contractive_factor:
            #we will calculate the contraction factor P =WG 
            #we will use power method to calculate the largest eigenvalue of P=WG
            #for power method we will call the function : power_method_for_images
            #we will construc the argument dict args_dict for calling the function
            #we will add input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, eta
            args_dict_P = {'denoiser': denoiser, 'denoiser_kwargs': denoiser_kwargs, \
                'A_function': A_function, 'A_kwargs': A_kwargs, 'A_function_adjoint': A_function_adjoint, \
                'A_adjoint_kwargs': A_adjoint_kwargs, 'eta': eta}
            #if the denoiser is DSG_NLM, then we will use the function P_operator for f in the power method function
            #if its NLM, we will use f= D_half_P_D_half_inverse
            if denoiser.__name__ == 'DSG_NLM':
                function_spectral_norm = P_operator
                function_spectral_norm_adjoint = P_operator_adjoint
            elif denoiser.__name__ == 'NLM':
                function_spectral_norm = D_half_P_D_half_inverse
                function_spectral_norm_adjoint = D_half_P_D_half_inverse_adjoint
                #we need to add the keyword argument 'D_matrix' to the args_dict_P
                #we can get teh D_matrix by calling teh denoiser with an extra keyword argument 'return_D_matrix' = True
                #add the keyword argument 'return_D_matrix' = True to the denoiser_kwargs
                denoiser_kwargs['return_D_matrix'] = True
                #call the denoiser
                _, D_matrix = denoiser(denoiser_kwargs['guide_image'], **denoiser_kwargs)
                #drop the keyword argument 'return_D_matrix' = True from the denoiser_kwargs dictionary
                del denoiser_kwargs['return_D_matrix']
                #add the D_matrix to the args_dict_P
                args_dict_P['D_matrix'] = D_matrix

            else:
                #assert error
                assert False, 'denoiser is not implemented'
            #call the power method function
            contractive_factor = power_method_for_images(f = function_spectral_norm, input_image = x_k, \
                args_dict = args_dict_P, max_iterations=power_method_max_iterations )
            # contractive_factor = power_method_for_images_non_symmetric(functional = function_spectral_norm, functional_adjoint = function_spectral_norm_adjoint, \
                # image_height = x_k.shape[0], image_width = x_k.shape[1], args_dict_functional = args_dict_P, \
                #  args_dict_functional_adjoint = args_dict_P, max_iterations=power_method_max_iterations )   
            # #append the contractive factor to the list
            # contractive_factor_list.append(contractive_factor)
            #print the contractive factor
            print('contractive factor: ', contractive_factor)
            #contractive factor ends ----------------------------------------------------------------------

        # #check that the keword argumnt in dictionary 'guide_image' is present and not None
        # assert 'guide_image' in denoiser_kwargs.keys() and denoiser_kwargs['guide_image'] is not None
        #denoise the update
        x_k_plus_1 = denoiser(grad_descent, **denoiser_kwargs)
        print(x_k_plus_1)
        #Gradient Updated ---------------------------------------------

        if calculate_contractive_factor:
            return x_k_plus_1, contractive_factor
        return x_k_plus_1

###########################################     ISTA  ENDS   #########################################################
####################################################################################################################


###########################################     ADMM     #########################################################
####################################################################################################################


def conjugate_gradient(A_function, A_kwargs, b, x0, max_iter , tol):
        """
        Conjugate gradient method for solving Ax=b
        """
        x = x0
        r = b-A_function(x, **A_kwargs)
        d = r
        for _ in range(max_iter):
            z = A_function(d, **A_kwargs)
            rr = np.sum(r**2)
            alpha = rr/np.sum(d*z)
            x += alpha*d
            r -= alpha*z
            if np.linalg.norm(r)/np.linalg.norm(b) < tol:
                break
            beta = np.sum(r**2)/rr
            d = r + beta*d        
        return x

def cg_leftside(x, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, step_size):
    """
    Return left side of Ax=b, i.e., Ax
    """
    return A_function_adjoint(A_function(x, **A_kwargs), **A_adjoint_kwargs) + step_size*x

def cg_rightside(x, b, step_size):
    """
    Returns right side of Ax=b, i.e. b
    """
    return b + step_size*x


#we define B _function 
#the function would do : x + step_size * A_function_adjoint(A_function(x, **A_kwargs), **A_adjoint_kwargs)
def B_function(x, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, step_size):
    """
    Return left side of Ax=b, i.e., Ax
    """
    return step_size*A_function_adjoint(A_function(x, **A_kwargs), **A_adjoint_kwargs) + x
#we define U_function as : input_image + step_size * A_function_adjoint(input_image, **A_adjoint_kwargs)
def U_function(x , b_H, step_size):
    """
    Return right side of Ax=b, i.e., b
    """
    return step_size*b_H + x



#we will flip the order of updates in the ADMM, inputs and output be same
#input: y__k, x_k, z_k, b_h, denoiser, denoiser_kwargs, iterations_to_fix_W, i, image
#output: y_k_plus_1, x_k_plus_1, z_k_plus_1
def ADMM(y_k, x_k, z_k, b_h, denoiser, denoiser_kwargs, cg_leftside_arg_dict, iterations_to_fix_W, \
         i, image, max_iter , tol):
    
    """ y_k+1 = denoiser(x_k - z_k, **denoiser_kwargs) """
    """ DENOISER """
    #we will calculate y_k_plus_1 as: y_k_plus_1 = denoiser(x_k - z_k, **denoiser_kwargs)
    #for denoiser we will ensure guide image

    #denoiser ----------------
    #add one keyword parameter to the denoiser "guide_image" , value grad_descent
    #this we will use to fix the guide image in the denoiser, i.e. parameter to the denoiser
    #we have if iterations_to_fix_W = -1, we will use the original image as the guide image
    #if denoiser is either DSG_NLM or NLM, i.e name is either 'DSG_NLM' or 'NLM'
    if denoiser.__name__ == 'DSG_NLM' or denoiser.__name__ == 'NLM':
        if iterations_to_fix_W == 0:
            denoiser_kwargs['guide_image'] = image
        #if iterations_to_fix_W is nay value > 0, we will adapt the guide image
        elif iterations_to_fix_W > 0:
            #current iteartion < iterations_to_fix_W, we will use the gradient
            if i < iterations_to_fix_W:
                denoiser_kwargs['guide_image'] = x_k - z_k
        #anything else will be error
        else:
            raise ValueError('iterations_to_fix_W should be a non-negative integer')
        #denoiser ----------------
        #check that the keword argumnt in dictionary 'guide_image' is present and not None
        assert 'guide_image' in denoiser_kwargs.keys() and denoiser_kwargs['guide_image'] is not None

    #else if denoiser is denoise_tv_chambolle
    #estimate sigma, and add weight parameter
    elif denoiser.__name__ == 'denoise_tv_chambolle':
        # #estimate sigma
        # sigma = estimate_sigma(x_k - z_k, multichannel=False)
        # #add weight parameter
        # denoiser_kwargs['weight'] = 0.1*sigma

        denoiser_kwargs['weight'] = 0.1

        #debug ----------------
        #print weight
        print('weight: ', denoiser_kwargs['weight'])
        #print max and min of x_k-z_k
        print('max of x_k-z_k: ', np.max(x_k-z_k))
        print('min of x_k-z_k: ', np.min(x_k-z_k))

        #max and min of x_k
        print('max of x_k: ', np.max(x_k))
        print('min of x_k: ', np.min(x_k))

        #max and min of z_k
        print('max of z_k: ', np.max(z_k))
        print('min of z_k: ', np.min(z_k))
        #debug ----------------

    #assert that anything else is error, denoiser not implamented
    else:
        raise ValueError('Denoiser not implemented')
    
    

    y_k_plus_1 = denoiser(x_k - z_k, **denoiser_kwargs)

    """ x_k+1 : Bx=u"""
    """ PROXIMAL OPERATOR OF F"""
    #we will get x_k_plus_1 as a solution of Bx=u
    #where, we will solve Bx=u by conjugate gradient method
    temp = U_function(x = y_k_plus_1 + z_k, b_H = b_h, step_size=cg_leftside_arg_dict['step_size'])
    x_k_plus_1 = conjugate_gradient(B_function, cg_leftside_arg_dict, temp, x_k.copy(), max_iter=max_iter,\
                                    tol = tol)

    """ Update variables. """
    z_k_plus_1 = z_k + y_k_plus_1 - x_k_plus_1

    #return the updated variables
    return y_k_plus_1, x_k_plus_1, z_k_plus_1


###########################################     ADMM ENDS     #########################################################
####################################################################################################################


