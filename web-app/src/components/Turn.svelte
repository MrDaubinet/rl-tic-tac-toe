<script>
  // Data props
  export let turn 
  export let agent_play = false
</script>

<div class="h-full rounded min-w-md">
  <div class="flex justify-between text-lg md:text-3xl text-white">
    <div class="{turn == 1 ? 'animate-pulse': ''}">
      <span class="ml-2">
        {#if agent_play}
          Player
        {:else}
          Player 1
        {/if} 
      </span>
    </div>
    <div class="{turn == 0 ? 'animate-pulse': ''}">
      <span class="ml-2 ">
        {#if agent_play}
          Agent
        {:else}
          Player 2
        {/if}
      </span>
    </div>
  </div>
</div> 