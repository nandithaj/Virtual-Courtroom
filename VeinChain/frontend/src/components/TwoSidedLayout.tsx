import * as React from 'react';
import AspectRatio from '@mui/joy/AspectRatio';
import Box from '@mui/joy/Box';
import Container from '@mui/joy/Container';
import { typographyClasses } from '@mui/joy/Typography';
import icon from "../assets/freepik__a-man-with-black-hair-around-25-years-old-sitting-__87611.png";
export default function TwoSidedLayout({
  children,
  reversed,
}: React.PropsWithChildren<{ reversed?: boolean }>) {
  return (
    <Container
      sx={[
        (theme) => ({
          position: 'relative',
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          py: 10,
          gap: 4,
          [theme.breakpoints.up(834)]: {
            flexDirection: 'row',
            gap: 6,
          },
          [theme.breakpoints.up(1199)]: {
            gap: 12,
          },
        }),
        reversed ? { flexDirection: 'column-reverse' } : { flexDirection: 'column' },
      ]}
    >
      <Box
        sx={(theme) => ({
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1rem',
          maxWidth: '50ch',
          textAlign: 'center',
          flexShrink: 999,
          [theme.breakpoints.up(834)]: {
            minWidth: 420,
            alignItems: 'flex-start',
            textAlign: 'initial',
          },
          [`& .${typographyClasses.root}`]: {
            textWrap: 'balance',
          },
        })}
      >
        {children}
      </Box>
      <AspectRatio
  ratio={600 / 520}
  maxHeight={300}
  sx={(theme) => ({
    minWidth: 300,
    alignSelf: 'stretch',
    [theme.breakpoints.up(834)]: {
      alignSelf: 'initial',
      flexGrow: 1,
      '--AspectRatio-maxHeight': '520px',
      '--AspectRatio-minHeight': '400px',
    },
    //border: 'none', // Remove any borders
    borderRadius: 'sm', // Ensure no rounded edges
    flexBasis: '50%',
    bgcolor: 'transparent', // Remove the blue/dark tint
    boxShadow: 'none', // Prevent any unwanted shadows
  })}
>
  <img src={icon} alt="" style={{ background: 'transparent' }} />
</AspectRatio>

    </Container>
  );
}