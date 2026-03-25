import random

while True:
    cuypuy_position = random.randint(1, 4)
    
    bentuk_goa = '|_|'
    goa_kosong = [bentuk_goa] * 4 
    goa = goa_kosong.copy() 
    goa[cuypuy_position - 1] = '|0_0|'

    print(f"\nCoba perhatikan goa ini: {' '.join(goa_kosong)}")

    # --- LOGIKA INPUT ---
    while True:
        pilihan_user = input('Pilih Goa yang ingin kamu masuki 1 / 2 / 3 / 4 : ')
        
        # Validasi: Cek apakah input kosong atau bukan angka
        if pilihan_user == '' or not pilihan_user.isdigit():
            print("Peringatan: Masukkan angka 1 sampai 4 saja!")
            continue
        
        pilihan_user = int(pilihan_user)
        
        # Cek apakah angka di antara 1 - 4
        if 1 <= pilihan_user <= 4:
            break
        else:
            print("Pilihan hanya tersedia 1, 2, 3, atau 4!")

    # --- CEK PEMENANG ---
    if pilihan_user == cuypuy_position:
        print(f"Selamat! Pilihan kamu benar! Cuypy berada di {' '.join(goa)}.")
    else:
        print(f"Maaf kamu kalah, Cuypy berada di {' '.join(goa)}, bukan di nomor {pilihan_user}.")

    # --- FITUR MAIN LAGI ---
    play_again = input('\nApakah anda ingin kembali bermain [y/n]: ').lower()
    if play_again == 'n':
        break

print('\nTerima kasih sudah bermain cuypy games!')