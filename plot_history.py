import pandas as pd
import os
import matplotlib.pyplot as plt
if __name__ == '__main__':
    base_dir = '/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc'
    hist_store_dir = os.path.join(base_dir,'hist_store')

    hist_df_origin = pd.read_csv(hist_store_dir+'/Origin_RetinaNet_512 (copy).csv')
    hist_df_customize = pd.read_csv(hist_store_dir+'/retinanet_tiled_512-new-.csv')
    fig,ax = plt.subplots(1,2,figsize = (10,8))

    epochs = range(len(hist_df_origin['loss']))
    epochs_adam = range(len(hist_df_customize['loss']))
    #--loss--
    loss = hist_df_origin['loss']
    val_loss = hist_df_origin['val_loss']
    loss_customize = hist_df_customize['loss']
    val_loss_customize = hist_df_customize['val_loss']
    #--plot--

    ax[0].plot(epochs,loss,label='train_loss_Origin_RetinaNet')
    ax[0].plot(epochs,val_loss,label='validation_loss_Origin_RetinaNet', color='orange', linestyle='-.')
    ax[0].set_title('Origin_RetinaNet')
    ax[0].legend()
    ax[1].plot(epochs_adam, loss_customize, label='train_loss_RetinaNet_Customize')
    ax[1].plot(epochs_adam, val_loss_customize, label='validation_loss_RetinaNet_Customize', color='orange', linestyle='-.')
    ax[1].set_title('Customize_RetinaNet')
    ax[1].legend()

    # --- Thêm giá trị cuối cùng cho SGD ---
    last_loss_Origin = loss.iloc[-1]
    last_val_loss_Origin = val_loss.iloc[-1]

    ax[0].annotate(f'{last_loss_Origin:.4f}', xy=(epochs[-1], last_loss_Origin),
                      textcoords="offset points", xytext=(5, 0), va='top', color='blue', fontsize=9)
    ax[0].annotate(f'{last_val_loss_Origin:.4f}', xy=(epochs[-1], last_val_loss_Origin),
                      textcoords="offset points", xytext=(5, 0), va='center', color='orange', fontsize=9)


    last_loss_customize = loss_customize.iloc[-1]
    last_val_loss_customize = val_loss_customize.iloc[-1]

    ax[1].annotate(f'{last_loss_customize:.4f}', xy=(epochs_adam[-1], last_loss_customize),
                      textcoords="offset points", xytext=(5, 0), va='top', color='blue', fontsize=9)
    ax[1].annotate(f'{last_val_loss_customize:.4f}', xy=(epochs_adam[-1], last_val_loss_customize),
                      textcoords="offset points", xytext=(5, 0), va='center', color='orange', fontsize=9)
    plt.tight_layout()
    plt.show()
