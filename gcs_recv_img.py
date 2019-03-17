import sys
import cv2
import ast

sys.path.append('..')
from VTOL.autonomy import setup_xbee

def main():
    # Instantiate XBee device
    device = setup_xbee()

    IMAGE_STRING_REP = None
    IMAGE_ARRAY = None
    FRAME_COUNTER = 0

    FRAME_SAVE_LOC = './new_frames/'

    try:
        device.flush_queues()
        print("Waiting for data...\n")
        
        while True: # NEEDS MODIFICATION TO LOOP TILL END OF FLIGHT MESSAGE IS RECIEVED

            while True: 
                xbee_message = device.read_data()
                if xbee_message is not None:

                    if xbee_message.data.decode() == "END OF IMAGE":
                        FRAME_COUNTER += 1
                        break;


                    IMAGE_STRING_REP = IMAGE_STRING_REP + xbee_message.data.decode()
                    
                else:
                    break

            IMAGE_ARRAY = ast.literal_eval(IMAGE_STRING_REP)


            img_path = FRAME_SAVE_LOC + 'frame' + str(FRAME_COUNTER) + '.jpg'

            cv2.imwrite(img_path, IMAGE_ARRAY)

            # NEEDS MODIFICATION TO LOOP TILL END OF FLIGHT MESSAGE IS RECIEVED


    finally:
        if device is not None and device.is_open():
            device.close()


if __name__ == '__main__':
    main()
